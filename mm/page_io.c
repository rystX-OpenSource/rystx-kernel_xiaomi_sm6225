// SPDX-License-Identifier: GPL-2.0
/*
 *  linux/mm/page_io.c
 *
 *  Copyright (C) 1991, 1992, 1993, 1994  Linus Torvalds
 *
 *  Swap reorganised 29.12.95, 
 *  Asynchronous swapping added 30.12.95. Stephen Tweedie
 *  Removed race in async swapping. 14.4.1996. Bruno Haible
 *  Add swap of shared pages through the page cache. 20.2.1998. Stephen Tweedie
 *  Always use brw_page, life becomes simpler. 12 May 1998 Eric Biederman
 */

#include <linux/mm.h>
#include <linux/kernel_stat.h>
#include <linux/gfp.h>
#include <linux/pagemap.h>
#include <linux/swap.h>
#include <linux/bio.h>
#include <linux/swapops.h>
#include <linux/buffer_head.h>
#include <linux/writeback.h>
#include <linux/frontswap.h>
#include <linux/blkdev.h>
#include <linux/psi.h>
#include <linux/uio.h>
#include <linux/sched/task.h>
#include <linux/pgtable.h>
#include <linux/kfifo.h>
#include <linux/lru_marie.h>
/*
 * kcompressd() calls kthread_should_stop().  Upstream picks kthread.h up
 * transitively (via zswap.h -> ... in the 6.x header graph); no include
 * reachable from this tree's page_io.c provides it, so pull it in
 * explicitly rather than rely on a transitive edge that does not exist here.
 */
#include <linux/kthread.h>

#ifdef CONFIG_LRU_MARIE
/*
 * data_race() is a KCSAN annotation that only arrived in 5.8 (commit
 * d071e91361bb); it expands to its argument and emits no code.  Define it
 * locally rather than dropping it from the expressions below, so the reads
 * upstream marked as intentionally-racy stay marked.  mm/kfence/core.c:37
 * does the same thing in this tree.
 */
#ifndef data_race
#define data_race(x) (x)
#endif

/*
 * Counter consumed by the early-OOM gate in
 * mm/page_alloc.c:should_reclaim_retry. Declared in include/linux/swap.h.
 * Marie-only: omitted entirely under CONFIG_LRU_MARIE=n.
 */
atomic_long_t nr_swap_write_failed = ATOMIC_LONG_INIT(0);
#endif

static struct bio *get_swap_bio(gfp_t gfp_flags,
				struct page *page, bio_end_io_t end_io)
{
	int i, nr = hpage_nr_pages(page);
	struct bio *bio;

	bio = bio_alloc(gfp_flags, nr);
	if (bio) {
		struct block_device *bdev;

		bio->bi_iter.bi_sector = map_swap_page(page, &bdev);
		bio_set_dev(bio, bdev);
		bio->bi_end_io = end_io;

		for (i = 0; i < nr; i++)
			bio_add_page(bio, page + i, PAGE_SIZE, 0);
		VM_BUG_ON(bio->bi_iter.bi_size != PAGE_SIZE * nr);
	}
	return bio;
}

void end_swap_bio_write(struct bio *bio)
{
	struct page *page = bio_first_page_all(bio);

	if (bio->bi_status) {
		SetPageError(page);
		/*
		 * We failed to write the page out to swap-space.
		 * Re-dirty the page in order to avoid it being reclaimed.
		 * Also print a dire warning that things will go BAD (tm)
		 * very quickly.
		 *
		 * Also clear PG_reclaim to avoid rotate_reclaimable_page()
		 *
		 * Bump nr_swap_write_failed so the early-OOM gate in
		 * should_reclaim_retry can short-circuit the
		 * MAX_RECLAIM_RETRIES wait when the swap backend (most
		 * commonly ZRAM/frontswap zs_malloc, or a real disk error) has
		 * stopped accepting writes — anon reclaim is doomed in that
		 * state regardless of get_nr_swap_pages() reporting free
		 * entries. Marie-only signal; vanilla Legacy builds
		 * (lru_marie_enabled()=false) skip the counter bump so the
		 * baseline allocator sees vanilla retry behaviour.
		 */
#ifdef CONFIG_LRU_MARIE
		if (lru_marie_enabled())
			atomic_long_inc(&nr_swap_write_failed);
#endif
		set_page_dirty(page);
		pr_alert_ratelimited("Write-error on swap-device (%u:%u:%llu)\n",
			 MAJOR(bio_dev(bio)),
			 MINOR(bio_dev(bio)),
			 (unsigned long long)bio->bi_iter.bi_sector);
		ClearPageReclaim(page);
	}
	end_page_writeback(page);
	bio_put(bio);
}

#ifdef CONFIG_LRU_MARIE
/*
 * Multi-page variant of end_swap_bio_write() for coalesced swap bios.
 *
 * Upstream iterates with bio_for_each_folio_all(); this tree has no folios
 * and no bio_for_each_folio_all(), so it walks the bio_vec array with
 * bio_for_each_segment_all() instead.  A THP contributes one bvec per
 * subpage here (get_swap_bio() adds them individually) rather than one
 * folio-sized bvec, so the writeback-end below is driven off the head page
 * of each segment and skipped for tails -- PG_writeback is PF_NO_TAIL, so
 * end_page_writeback() must only be called once per compound page.
 *
 * Unlike end_swap_bio_write() this does NOT bio_put(); the batch submitter
 * owns the bio because it used submit_bio_wait().
 */
static void __end_swap_bio_write_batch(struct bio *bio)
{
	struct bio_vec *bv;
	int i;

	if (bio->bi_status) {
		bio_for_each_segment_all(bv, bio, i) {
			struct page *page = bv->bv_page;

			if (PageTail(page))
				continue;
			if (lru_marie_enabled())
				atomic_long_inc(&nr_swap_write_failed);
			SetPageError(page);
			set_page_dirty(page);
			ClearPageReclaim(page);
		}
		pr_alert_ratelimited("Write-error on swap-device (%u:%u:%llu)\n",
			MAJOR(bio_dev(bio)), MINOR(bio_dev(bio)),
			(unsigned long long)bio->bi_iter.bi_sector);
	}
	bio_for_each_segment_all(bv, bio, i) {
		if (PageTail(bv->bv_page))
			continue;
		end_page_writeback(bv->bv_page);
	}
}
#endif /* CONFIG_LRU_MARIE */

static void end_swap_bio_read(struct bio *bio)
{
	struct page *page = bio_first_page_all(bio);
	struct task_struct *waiter = bio->bi_private;

	if (bio->bi_status) {
		SetPageError(page);
		ClearPageUptodate(page);
		pr_alert("Read-error on swap-device (%u:%u:%llu)\n",
			 MAJOR(bio_dev(bio)), MINOR(bio_dev(bio)),
			 (unsigned long long)bio->bi_iter.bi_sector);
		goto out;
	}

	SetPageUptodate(page);
out:
	unlock_page(page);
	WRITE_ONCE(bio->bi_private, NULL);
	bio_put(bio);
	wake_up_process(waiter);
	put_task_struct(waiter);
}

int generic_swapfile_activate(struct swap_info_struct *sis,
				struct file *swap_file,
				sector_t *span)
{
	struct address_space *mapping = swap_file->f_mapping;
	struct inode *inode = mapping->host;
	unsigned blocks_per_page;
	unsigned long page_no;
	unsigned blkbits;
	sector_t probe_block;
	sector_t last_block;
	sector_t lowest_block = -1;
	sector_t highest_block = 0;
	int nr_extents = 0;
	int ret;

	blkbits = inode->i_blkbits;
	blocks_per_page = PAGE_SIZE >> blkbits;

	/*
	 * Map all the blocks into the extent list.  This code doesn't try
	 * to be very smart.
	 */
	probe_block = 0;
	page_no = 0;
	last_block = i_size_read(inode) >> blkbits;
	while ((probe_block + blocks_per_page) <= last_block &&
			page_no < sis->max) {
		unsigned block_in_page;
		sector_t first_block;

		cond_resched();

		first_block = probe_block;
		ret = bmap(inode, &first_block);
		if (ret || !first_block)
			goto bad_bmap;

		/*
		 * It must be PAGE_SIZE aligned on-disk
		 */
		if (first_block & (blocks_per_page - 1)) {
			probe_block++;
			goto reprobe;
		}

		for (block_in_page = 1; block_in_page < blocks_per_page;
					block_in_page++) {
			sector_t block;

			block = probe_block + block_in_page;
			ret = bmap(inode, &block);
			if (ret || !block)
				goto bad_bmap;

			if (block != first_block + block_in_page) {
				/* Discontiguity */
				probe_block++;
				goto reprobe;
			}
		}

		first_block >>= (PAGE_SHIFT - blkbits);
		if (page_no) {	/* exclude the header page */
			if (first_block < lowest_block)
				lowest_block = first_block;
			if (first_block > highest_block)
				highest_block = first_block;
		}

		/*
		 * We found a PAGE_SIZE-length, PAGE_SIZE-aligned run of blocks
		 */
		ret = add_swap_extent(sis, page_no, 1, first_block);
		if (ret < 0)
			goto out;
		nr_extents += ret;
		page_no++;
		probe_block += blocks_per_page;
reprobe:
		continue;
	}
	ret = nr_extents;
	*span = 1 + highest_block - lowest_block;
	if (page_no == 0)
		page_no = 1;	/* force Empty message */
	sis->max = page_no;
	sis->pages = page_no - 1;
	sis->highest_bit = page_no - 1;
out:
	return ret;
bad_bmap:
	pr_err("swapon: swapfile has holes\n");
	ret = -EINVAL;
	goto out;
}

/*
 * do_swapout() - Write a page to swap space
 * @page: The page to write out
 *
 * This function writes the page to swap space, either using frontswap or
 * synchronous write. It ensures that the page is unlocked and the
 * reference count is decremented after the operation.
 *
 * Upstream's zswap_store() is spelled frontswap_store() here: this tree
 * predates zswap's promotion to a first-class swap backend, and
 * frontswap_store() returning 0 is the "the backend took it, no bio
 * needed" case that zswap_store() returning true denotes upstream.  The
 * writeback bracketing around it mirrors swap_writepage() below.
 * count_mthp_stat() has no analogue in this tree (per-order THP swapout
 * stats arrived in 6.10) and is dropped.
 */
static inline void do_swapout(struct page *page, struct writeback_control *wbc)
{
	if (frontswap_store(page) == 0) {
		set_page_writeback(page);
		unlock_page(page);
		end_page_writeback(page);
	} else
		/* Implies unlock_page(page) */
		__swap_writepage(page, wbc, end_swap_bio_write);

	/* Decrement the page reference count */
	put_page(page);
}

#ifdef CONFIG_LRU_MARIE
/* Forward decl: defined below, after __swap_writepage(). */
static void swap_writepage_bdev_sync_batch(struct page **pages,
		unsigned int n, struct swap_info_struct *sis,
		struct writeback_control *wbc);

/*
 * do_swapout_batch() - Write a drained batch of pages to swap.
 *
 * Partitions the batch into runs eligible for bio-coalescing (same
 * swap_info_struct, all anon, frontswap off, SWP_SYNCHRONOUS_IO) and
 * dispatches each such run through swap_writepage_bdev_sync_batch() --
 * one multi-segment bio per contiguous-slot run instead of one bio per
 * page. Pages that don't qualify (frontswap on, non-anon, non-sync-IO
 * device) fall back to the per-page do_swapout() path unchanged.
 */
static void do_swapout_batch(struct page **pages, unsigned int n,
			     struct writeback_control *wbc)
{
	unsigned int i = 0;

	while (i < n) {
		struct page *p = pages[i];
		struct swap_info_struct *sis = page_swap_info(p);

		/* frontswap on, non-anon, or non-sync-IO block dev -> per-page. */
		if (frontswap_enabled() || !PageAnon(p) ||
		    !data_race(sis->flags & SWP_SYNCHRONOUS_IO)) {
			do_swapout(p, wbc);
			i++;
			continue;
		}

		/* Build a same-sis run. */
		{
			unsigned int start = i;
			unsigned int j;

			while (i < n) {
				struct page *g = pages[i];

				if (page_swap_info(g) != sis ||
				    !PageAnon(g) ||
				    frontswap_enabled())
					break;
				i++;
			}
			swap_writepage_bdev_sync_batch(&pages[start], i - start,
						       sis, wbc);
			/* drop the refs kcompressd_store took (do_swapout would). */
			for (j = start; j < i; j++)
				put_page(pages[j]);
		}
	}
}

/*
 * kcompressd_store() - Off-load page compression to kcompressd
 * @page: The page to compress
 *
 * This function attempts to off-load the compression of the page to
 * kcompressd. If kcompressd is not available or the page cannot be
 * compressed, it falls back to synchronous write.
 *
 * Returns true if the page was successfully queued for compression,
 * false otherwise.
 */
static bool kcompressd_store(struct page *page, struct writeback_control *wbc)
{
	pg_data_t *pgdat = NODE_DATA(numa_node_id());
	unsigned int ret;
	struct page *head = NULL;
	unsigned long flags;

	/* Only kswapd can use kcompressd */
	if (!current_is_kswapd())
		return false;

	/* Mode 0, or mode 1 with Marie off — short-circuit on the static branches. */
	if (!kcompressd_active())
		return false;

	/* kthread must be running */
	if (unlikely(!pgdat->kcompressd))
		return false;

	/* We can only off-load anon pages */
	if (!PageAnon(page))
		return false;

	/*
	 * Upstream additionally refuses the off-load when the folio's memcg
	 * has zswap writeback disabled (memory.zswap.writeback, 6.8 commit
	 * 501a06fe8e4c), taking rcu_read_lock() around folio_memcg() for the
	 * obj_cgroup_memcg lockdep assert.  Neither the knob nor the objcg
	 * indirection exists in this tree, so the check has no analogue and
	 * is dropped; frontswap has no per-memcg writeback policy to honour.
	 */

	/* Swap device must be sync-efficient */
	if (!frontswap_enabled() &&
		!data_race(page_swap_info(page)->flags & SWP_SYNCHRONOUS_IO))
		return false;

	/*
	 * The kfifo backing storage is sized at KCOMPRESSD_FIFO_SIZE (the
	 * compile-time max). The effective queue depth is |vm_kcompressd|;
	 * when current depth meets or exceeds that, treat the queue as
	 * full and swap out the head page synchronously to make space.
	 *
	 * Upstream wraps this in scoped_guard(spinlock_irqsave, ...), a
	 * cleanup.h construct that only arrived in 6.7 (commit 54da6a092431);
	 * the explicit lock/unlock pair below is the same critical section,
	 * with the guard's implicit unlock on the early return spelled out.
	 */
	spin_lock_irqsave(&pgdat->kcompressd_fifo_lock, flags);
	if (kfifo_len(&pgdat->kcompressd_fifo) >=
		abs(READ_ONCE(vm_kcompressd)) * sizeof(struct page *) &&
		unlikely(!kfifo_out(&pgdat->kcompressd_fifo,
				&head, sizeof(page)))) {
		spin_unlock_irqrestore(&pgdat->kcompressd_fifo_lock, flags);
		return false;
	}
	spin_unlock_irqrestore(&pgdat->kcompressd_fifo_lock, flags);

	/* Increment the page reference count to avoid it being freed */
	get_page(page);

	/* Enqueue the page for compression */
	ret = kfifo_in(&pgdat->kcompressd_fifo, &page, sizeof(page));
	if (likely(ret))
		/* We successfully enqueued the page. wake up kcompressd */
		wake_up_interruptible(&pgdat->kcompressd_wait);
	else
		/* Enqueue failed, so we must cancel the reference count */
		put_page(page);

	/* If we had to swap out the head page, do it now.
	 * This will block until the page is written out.
	 */
	if (head)
		do_swapout(head, wbc);

	return ret;
}
#else  /* !CONFIG_LRU_MARIE */
static inline bool kcompressd_store(struct page *page,
				   struct writeback_control *wbc)
{
	return false;
}
#endif

/*
 * We may have stale swap cache pages in memory: notice
 * them here and get rid of the unnecessary final write.
 */
int swap_writepage(struct page *page, struct writeback_control *wbc)
{
	int ret = 0;

	if (try_to_free_swap(page)) {
		unlock_page(page);
		goto out;
	}

	/*
	 * Compression within zswap and zram might block rmap, unmap
	 * of both file and anon pages, try to do compression async
	 * if possible
	 */
	if (kcompressd_store(page, wbc))
		return 0;

	if (frontswap_store(page) == 0) {
		set_page_writeback(page);
		unlock_page(page);
		end_page_writeback(page);
		goto out;
	}
	ret = __swap_writepage(page, wbc, end_swap_bio_write);
out:
	return ret;
}

#ifdef CONFIG_LRU_MARIE
/*
 * Batch-drain size for kcompressd(): reuses the existing |vm_kcompressd|
 * queue-depth knob (no separate batch-size knob) clamped to a sane range
 * for the stack-allocated drain array.
 */
static inline unsigned int kcompressd_drain_depth(void)
{
	int d = abs(READ_ONCE(vm_kcompressd));

	if (d < 1)
		return 1;
	if (d > SWAP_CLUSTER_MAX)
		return SWAP_CLUSTER_MAX;
	return d;
}

/*
 * kcompressd() - Kernel thread for compressing pages
 * @p: Pointer to pg_data_t structure
 *
 * This function runs in a kernel thread and waits for pages to be
 * queued for compression. It drains its fifo in batches and dispatches
 * each batch through do_swapout_batch(), which coalesces same-device
 * contiguous-slot pages into multi-segment bios (see Stage 1 above).
 */
int kcompressd(void *p)
{
	pg_data_t *pgdat = (pg_data_t *)p;
	struct page *batch[SWAP_CLUSTER_MAX];
	unsigned int n;
	/*
	 * do_swapout -> __swap_writepage dereferences wbc via
	 * wbc_to_write_flags(); passing NULL faults at wbc->sync_mode.
	 * kcompressd is async reclaim writeback, so a stack wbc with
	 * WB_SYNC_NONE matches the semantics and keeps the writepage paths
	 * from having to special-case NULL.  (Upstream also zeroes
	 * .swap_plug, a field this tree does not have.)
	 */
	struct writeback_control wbc = {
		.sync_mode = WB_SYNC_NONE,
	};
	/* * kcompressd runs with PF_MEMALLOC and PF_KSWAPD flags set to
	 * allow it to allocate memory for compression without being
	 * restricted by the current memory allocation context.
	 * Also PF_KSWAPD prevents Intel Graphics driver from crashing
	 * the system in i915_gem_shrinker.c:i915_gem_shrinker_scan()
	 */
	current->flags |= PF_MEMALLOC | PF_KSWAPD;

	while (!kthread_should_stop()) {
		wait_event_interruptible(pgdat->kcompressd_wait,
				!kfifo_is_empty(&pgdat->kcompressd_fifo));

		for (;;) {
			n = kfifo_out_locked(&pgdat->kcompressd_fifo,
				batch,
				kcompressd_drain_depth() * sizeof(*batch),
				&pgdat->kcompressd_fifo_lock)
					/ sizeof(*batch);
			if (!n)
				break;
			do_swapout_batch(batch, n, &wbc);
		}
	}
	return 0;
}
#endif /* CONFIG_LRU_MARIE */

static inline void count_swpout_vm_event(struct page *page)
{
#ifdef CONFIG_TRANSPARENT_HUGEPAGE
	if (unlikely(PageTransHuge(page)))
		count_vm_event(THP_SWPOUT);
#endif
	count_vm_events(PSWPOUT, hpage_nr_pages(page));
}

int __swap_writepage(struct page *page, struct writeback_control *wbc,
		bio_end_io_t end_write_func)
{
	struct bio *bio;
	int ret;
	struct swap_info_struct *sis = page_swap_info(page);

	VM_BUG_ON_PAGE(!PageSwapCache(page), page);
	if (sis->flags & SWP_FILE) {
		struct kiocb kiocb;
		struct file *swap_file = sis->swap_file;
		struct address_space *mapping = swap_file->f_mapping;
		struct bio_vec bv = {
			.bv_page = page,
			.bv_len  = PAGE_SIZE,
			.bv_offset = 0
		};
		struct iov_iter from;

		iov_iter_bvec(&from, ITER_BVEC | WRITE, &bv, 1, PAGE_SIZE);
		init_sync_kiocb(&kiocb, swap_file);
		kiocb.ki_pos = page_file_offset(page);

		set_page_writeback(page);
		unlock_page(page);
		ret = mapping->a_ops->direct_IO(&kiocb, &from);
		if (ret == PAGE_SIZE) {
			count_vm_event(PSWPOUT);
			ret = 0;
		} else {
			/*
			 * In the case of swap-over-nfs, this can be a
			 * temporary failure if the system has limited
			 * memory for allocating transmit buffers.
			 * Mark the page dirty and avoid
			 * rotate_reclaimable_page but rate-limit the
			 * messages but do not flag PageError like
			 * the normal direct-to-bio case as it could
			 * be temporary.
			 */
			set_page_dirty(page);
			ClearPageReclaim(page);
			pr_err_ratelimited("Write error on dio swapfile (%llu)\n",
					   page_file_offset(page));
		}
		end_page_writeback(page);
		return ret;
	}

	ret = bdev_write_page(sis->bdev, map_swap_page(page, &sis->bdev),
			      page, wbc);
	if (!ret) {
		count_swpout_vm_event(page);
		return 0;
	}

	ret = 0;
	bio = get_swap_bio(GFP_NOIO, page, end_write_func);
	if (bio == NULL) {
		set_page_dirty(page);
		unlock_page(page);
		ret = -ENOMEM;
		goto out;
	}
	bio->bi_opf = REQ_OP_WRITE | REQ_SWAP | wbc_to_write_flags(wbc);
	bio_associate_blkcg_from_page(bio, page);
	count_swpout_vm_event(page);
	set_page_writeback(page);
	unlock_page(page);
	submit_bio(bio);
out:
	return ret;
}

#ifdef CONFIG_LRU_MARIE
/*
 * Stage 1: coalesce a run of pages with *physically contiguous* swap slots
 * into one multi-segment bio and submit_bio_wait() it once. Non-contiguous
 * pages start a new bio (graceful fallback to the per-page granularity).
 *
 * Two adaptations from upstream, both forced by this tree's bio API:
 *
 *  - upstream sizes the bio in folios (one bio_add_folio_nofail() per
 *    folio, capped at BIO_MAX_VECS).  Here a THP contributes one bvec per
 *    subpage, exactly as get_swap_bio() builds it, so the segment budget
 *    counts pages and is capped at BIO_MAX_PAGES.
 *
 *  - upstream reads the slot with swap_folio_sector(); this tree's
 *    equivalent is map_swap_page(), which also hands back the backing
 *    block_device (there is no bio_alloc(bdev, ...) to pass it to, so it
 *    goes in via bio_set_dev()).
 */
static void swap_writepage_bdev_sync_batch(struct page **pages,
		unsigned int n, struct swap_info_struct *sis,
		struct writeback_control *wbc)
{
	unsigned int i = 0;

	while (i < n) {
		struct bio *bio;
		struct block_device *bdev;
		sector_t first, next;
		unsigned int start = i, segs = 0, j;

		first = next = map_swap_page(pages[i], &bdev);

		/* Grow the contiguous run [start, i). */
		while (i < n) {
			sector_t sect = map_swap_page(pages[i], &bdev);
			unsigned int nr = hpage_nr_pages(pages[i]);

			if (sect != next || segs + nr > BIO_MAX_PAGES)
				break;
			next += (nr << (PAGE_SHIFT - 9));
			segs += nr;
			i++;
		}

		bio = bio_alloc(GFP_NOIO, segs);
		if (unlikely(!bio)) {
			/*
			 * Same recovery as __swap_writepage()'s ENOMEM path:
			 * redirty so reclaim retries, and unlock since the
			 * caller handed these pages over locked.
			 */
			for (j = start; j < i; j++) {
				set_page_dirty(pages[j]);
				unlock_page(pages[j]);
			}
			continue;
		}
		bio_set_dev(bio, bdev);
		bio->bi_iter.bi_sector = first;
		bio->bi_opf = REQ_OP_WRITE | REQ_SWAP | wbc_to_write_flags(wbc);
		bio_associate_blkcg_from_page(bio, pages[start]);
		for (j = start; j < i; j++) {
			unsigned int k, nr = hpage_nr_pages(pages[j]);

			count_swpout_vm_event(pages[j]);
			set_page_writeback(pages[j]);
			unlock_page(pages[j]);
			for (k = 0; k < nr; k++)
				bio_add_page(bio, pages[j] + k, PAGE_SIZE, 0);
		}
		submit_bio_wait(bio);
		__end_swap_bio_write_batch(bio);
		bio_put(bio);
	}
}
#endif /* CONFIG_LRU_MARIE */

int swap_readpage(struct page *page, bool synchronous)
{
	struct bio *bio;
	int ret = 0;
	struct swap_info_struct *sis = page_swap_info(page);
	blk_qc_t qc;
	struct gendisk *disk;
	unsigned long pflags;

	VM_BUG_ON_PAGE(!PageSwapCache(page) && !synchronous, page);
	VM_BUG_ON_PAGE(!PageLocked(page), page);
	VM_BUG_ON_PAGE(PageUptodate(page), page);

	/*
	 * Count submission time as memory stall. When the device is congested,
	 * or the submitting cgroup IO-throttled, submission can be a
	 * significant part of overall IO time.
	 */
	psi_memstall_enter(&pflags);

	if (frontswap_load(page) == 0) {
		SetPageUptodate(page);
		unlock_page(page);
		goto out;
	}

	if (sis->flags & SWP_FILE) {
		struct file *swap_file = sis->swap_file;
		struct address_space *mapping = swap_file->f_mapping;

		ret = mapping->a_ops->readpage(swap_file, page);
		if (!ret)
			count_vm_event(PSWPIN);
		goto out;
	}

	ret = bdev_read_page(sis->bdev, map_swap_page(page, &sis->bdev), page);
	if (!ret) {
		count_vm_event(PSWPIN);
		goto out;
	}

	ret = 0;
	bio = get_swap_bio(GFP_KERNEL, page, end_swap_bio_read);
	if (bio == NULL) {
		unlock_page(page);
		ret = -ENOMEM;
		goto out;
	}
	disk = bio->bi_disk;
	/*
	 * Keep this task valid during swap readpage because the oom killer may
	 * attempt to access it in the page fault retry time check.
	 */
	get_task_struct(current);
	bio->bi_private = current;
	bio_set_op_attrs(bio, REQ_OP_READ, 0);
	count_vm_event(PSWPIN);
	bio_get(bio);
	qc = submit_bio(bio);
	while (synchronous) {
		set_current_state(TASK_UNINTERRUPTIBLE);
		if (!READ_ONCE(bio->bi_private))
			break;

		if (!blk_poll(disk->queue, qc))
			break;
	}
	__set_current_state(TASK_RUNNING);
	bio_put(bio);

out:
	psi_memstall_leave(&pflags);
	return ret;
}

int swap_set_page_dirty(struct page *page)
{
	struct swap_info_struct *sis = page_swap_info(page);

	if (sis->flags & SWP_FILE) {
		struct address_space *mapping = sis->swap_file->f_mapping;

		VM_BUG_ON_PAGE(!PageSwapCache(page), page);
		return mapping->a_ops->set_page_dirty(page);
	} else {
		return __set_page_dirty_no_writeback(page);
	}
}
