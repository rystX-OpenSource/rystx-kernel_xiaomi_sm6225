#ifndef _LINUX_MMAP_LOCK_H
#define _LINUX_MMAP_LOCK_H

static inline void mmap_init_lock(struct mm_struct *mm)
{
	init_rwsem(&mm->mmap_lock);
}

static inline void mmap_write_lock(struct mm_struct *mm)
{
	down_write(&mm->mmap_lock);
}

static inline int mmap_write_lock_killable(struct mm_struct *mm)
{
	return down_write_killable(&mm->mmap_lock);
}

static inline bool mmap_write_trylock(struct mm_struct *mm)
{
	return down_write_trylock(&mm->mmap_lock) != 0;
}

static inline void mmap_write_unlock(struct mm_struct *mm)
{
	up_write(&mm->mmap_lock);
}

static inline void mmap_write_downgrade(struct mm_struct *mm)
{
	downgrade_write(&mm->mmap_lock);
}

static inline void mmap_read_lock(struct mm_struct *mm)
{
	down_read(&mm->mmap_lock);
}

static inline int mmap_read_lock_killable(struct mm_struct *mm)
{
	return down_read_killable(&mm->mmap_lock);
}

static inline bool mmap_read_trylock(struct mm_struct *mm)
{
	return down_read_trylock(&mm->mmap_lock) != 0;
}

static inline void mmap_read_unlock(struct mm_struct *mm)
{
	up_read(&mm->mmap_lock);
}

static inline bool mmap_read_trylock_non_owner(struct mm_struct *mm)
{
	if (down_read_trylock(&mm->mmap_lock)) {
		rwsem_release(&mm->mmap_lock.dep_map, _RET_IP_);
		return true;
	}
	return false;
}

static inline void mmap_read_unlock_non_owner(struct mm_struct *mm)
{
	up_read_non_owner(&mm->mmap_lock);
}

static inline int mmap_lock_is_contended(struct mm_struct *mm)
{
	return rwsem_is_contended(&mm->mmap_lock);
}

/*
 * mmap_assert_locked() / mmap_assert_write_locked() -- hard-copied from 6.19.8
 * include/linux/mmap_lock.h:69-77 (upstream commit 42fc541404f2 "mmap_lock: add
 * mmap_assert_locked() and mmap_assert_write_locked()").
 *
 * Upstream's one-line bodies delegate to rwsem_assert_held{,_write}(), which
 * this tree does not have.  Those are themselves only a lockdep/no-lockdep
 * either-or (cachy include/linux/rwsem.h:204-215):
 *
 *	if (IS_ENABLED(CONFIG_LOCKDEP))
 *		lockdep_assert_held(sem);
 *	else
 *		rwsem_assert_held_nolockdep(sem);
 *
 * so that shape is reproduced inline here rather than growing rwsem.h for two
 * callers.  With lockdep on this is the exact upstream assertion; with lockdep
 * off it degrades to the same is-it-locked-at-all check upstream degrades to
 * (rwsem_assert_held_nolockdep() is a count != RWSEM_UNLOCKED_VALUE test, which
 * is precisely this tree's rwsem_is_locked(), rwsem.h:68).
 *
 * The write variant cannot narrow to write-held without lockdep on this tree:
 * upstream's nolockdep spelling tests RWSEM_WRITER_LOCKED, a bit this tree's
 * asm-generic rwsem does not define (it predates the 5.3 rwsem rewrite and uses
 * RWSEM_ACTIVE_WRITE_BIAS arithmetic instead).  Falling back to the weaker
 * locked-at-all test matches what mm/pagewalk.c:320 already asserts here and
 * keeps the debug-only helper from carrying arch-specific bias math.
 *
 * rwsem_is_locked() takes a non-const sem in this tree, hence the cast on the
 * otherwise-const upstream parameter.
 */
static inline void mmap_assert_locked(const struct mm_struct *mm)
{
	if (IS_ENABLED(CONFIG_LOCKDEP))
		lockdep_assert_held(&mm->mmap_lock);
	else
		WARN_ON(!rwsem_is_locked(
			(struct rw_semaphore *)&mm->mmap_lock));
}

static inline void mmap_assert_write_locked(const struct mm_struct *mm)
{
	if (IS_ENABLED(CONFIG_LOCKDEP))
		lockdep_assert_held_write(&mm->mmap_lock);
	else
		WARN_ON(!rwsem_is_locked(
			(struct rw_semaphore *)&mm->mmap_lock));
}

#endif /* _LINUX_MMAP_LOCK_H */
