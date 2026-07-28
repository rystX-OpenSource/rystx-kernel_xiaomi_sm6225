// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * zram backend for LZ4KDR (see include/linux/lz4kdr.h for what it is
 * and how it relates to Huawei's original LZ4KD).
 *
 * No dict/level support -- LZ4KDR doesn't have either concept, so
 * setup_params/release_params are no-ops, same as backend_842.c.
 *
 * The context's `ht` buffer is the hash table lz4kdr_encode() searches
 * with. It's kzalloc()'d once here (not vmalloc()'d -- at
 * lz4kdr_encode_state_bytes_min()==2048 bytes it's well within
 * kmalloc's normal range, and per-CPU vmalloc mappings only cost more
 * TLB pressure for no benefit at this size) and, critically, the
 * kzalloc zeroing IS the one-time cold-start zero lz4kdr_encode()'s
 * API contract requires -- lz4kdr does NOT re-zero this buffer on
 * every compress call the way upstream LZ4KD does, see
 * lz4kdr_encode.c's probe-loop comment for why that's still correct.
 */

#include <linux/kernel.h>
#include <linux/lz4kdr.h>
#include <linux/slab.h>

#include "backend_lz4kdr.h"

struct lz4kdr_ctx {
	void *ht;
};

static void lz4kdr_release_params(struct zcomp_params *params)
{
}

static int lz4kdr_setup_params(struct zcomp_params *params)
{
	return 0;
}

static void lz4kdr_destroy(struct zcomp_ctx *ctx)
{
	struct lz4kdr_ctx *zctx = ctx->context;

	if (!zctx)
		return;

	kfree(zctx->ht);
	kfree(zctx);
}

static int lz4kdr_create(struct zcomp_params *params, struct zcomp_ctx *ctx)
{
	struct lz4kdr_ctx *zctx;

	zctx = kzalloc(sizeof(*zctx), GFP_KERNEL);
	if (!zctx)
		return -ENOMEM;

	ctx->context = zctx;
	/*
	 * kzalloc, not kmalloc: the zeroing here IS the required one-time
	 * cold-start zero, see the top-of-file comment.
	 */
	zctx->ht = kzalloc(lz4kdr_encode_state_bytes_min(), GFP_KERNEL);
	if (!zctx->ht)
		goto error;

	return 0;

error:
	lz4kdr_destroy(ctx);
	return -ENOMEM;
}

static int lz4kdr_compress(struct zcomp_params *params, struct zcomp_ctx *ctx,
			   struct zcomp_req *req)
{
	struct lz4kdr_ctx *zctx = ctx->context;
	int ret;

	ret = lz4kdr_encode(zctx->ht, req->src, req->dst,
			    (unsigned int)req->src_len,
			    (unsigned int)req->dst_len, 0);
	if (ret < 0)
		return -EINVAL;
	if (ret == 0) {
		/*
		 * INCOMPRESSIBLE (not an error): lz4kdr bailed early because
		 * the page won't beat its out_limit. This is exactly the case
		 * zram handles by storing the page raw -- LZ4/zstd signal it by
		 * returning a comp_len >= huge_class_size, which they always
		 * can because their dst buffer is oversized; lz4kdr instead has
		 * an explicit "0 == did not compress" verdict. Report a
		 * full-page length so zram_write_page() takes its huge/raw-store
		 * path (comp_len == PAGE_SIZE), or, under CONFIG_ZRAM_MULTI_COMP
		 * / ZRAM-IR, retries the page with the next (higher-ratio)
		 * compressor.
		 *
		 * Returning -EINVAL here (as `ret <= 0` did) makes
		 * zcomp_compress() fail the whole write: zram_write_page() then
		 * bails with "Compression failed!" instead of storing the page,
		 * so reclaim can't evict it. Under memory pressure that stalls
		 * reclaim and fires the OOM killer the instant swapout hits its
		 * first incompressible page. req->src_len is PAGE_SIZE for zram
		 * writes, which is the sentinel the raw-store path keys on.
		 */
		req->dst_len = req->src_len;
		return 0;
	}
	req->dst_len = ret;
	return 0;
}

static int lz4kdr_decompress(struct zcomp_params *params, struct zcomp_ctx *ctx,
			     struct zcomp_req *req)
{
	int ret;

	ret = lz4kdr_decode(req->src, req->dst,
			    (unsigned int)req->src_len,
			    (unsigned int)req->dst_len);
	if (ret <= 0)
		return -EINVAL;
	return 0;
}

const struct zcomp_ops backend_lz4kdr = {
	.compress	= lz4kdr_compress,
	.decompress	= lz4kdr_decompress,
	.create_ctx	= lz4kdr_create,
	.destroy_ctx	= lz4kdr_destroy,
	.setup_params	= lz4kdr_setup_params,
	.release_params	= lz4kdr_release_params,
	.name		= "lz4kdr",
};