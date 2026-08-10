/* SPDX-License-Identifier: GPL-2.0 */
#ifndef __LINUX_GUARDS_H
#define __LINUX_GUARDS_H

/*
 * Scope-based resource management -- backported subset.
 *
 * Hard-copied from Linux 6.19.8 include/linux/cleanup.h (upstream series
 * "Scope-based Resource Management", first commit 54da6a092431) and reduced
 * to the machinery this tree actually has callers for:
 *
 *   DEFINE_CLASS() / CLASS()      -- constructor/destructor pairing
 *   DEFINE_GUARD() / guard()      -- unconditional lock guards
 *   DEFINE_LOCK_GUARD_1()         -- lock guards carrying extra state
 *   scoped_guard()                -- guard bound to a statement scope
 *
 * Deliberately omitted, having no in-tree users at this version: the
 * DEFINE_FREE()/no_free_ptr()/__free() allocation helpers, the _COND_
 * (conditional / try-lock) guard variants, ACQUIRE()/ACQUIRE_ERR(), and
 * DEFINE_LOCK_GUARD_0().  The _COND_ variants additionally need
 * typeof_member(), which does not exist in 4.19.  Add them here from the
 * same 6.19.8 source if a caller ever appears -- do not reinvent them.
 *
 * 4.19 adaptations:
 *   - __cleanup() is defined here; 4.19's compiler_attributes.h predates it.
 *   - __always_inline, __maybe_unused, __force, IS_ERR, ERR_PTR and
 *     MAX_ERRNO all already exist in this tree and are used unchanged.
 */

#include <linux/compiler.h>
#include <linux/err.h>

/*
 * GCC 4.9+ / clang: run the destructor when the variable leaves scope.
 * 4.19's minimum GCC is 4.6 for most arches but 4.9 in practice for the
 * arches this tree builds; the attribute has been available since 3.x.
 */
#ifndef __cleanup
#define __cleanup(func) __attribute__((__cleanup__(func)))
#endif

/*
 * DEFINE_CLASS(name, type, exit, init, init_args...):
 *
 *	class_##name##_t		// the type
 *	class_##name##_constructor()	// runs @init, returns the type
 *	class_##name##_destructor()	// runs @exit with _T bound to the value
 *
 * CLASS(name, var)(args...) declares @var of that type, initialised by the
 * constructor and destroyed on scope exit.
 */
#define DEFINE_CLASS(_name, _type, _exit, _init, _init_args...)		\
typedef _type class_##_name##_t;					\
static __always_inline void class_##_name##_destructor(_type *p)	\
{ _type _T = *p; _exit; }						\
static __always_inline _type class_##_name##_constructor(_init_args)	\
{ _type t = _init; return t; }

#define CLASS(_name, var)						\
	class_##_name##_t var __cleanup(class_##_name##_destructor) =	\
		class_##_name##_constructor

/*
 * Guards are classes whose value is a lock pointer.  class_*_lock_ptr()
 * reports whether the lock was actually acquired; for the unconditional
 * guards backported here it is always "yes", but scoped_guard() still
 * consults it so the conditional variants can share the same expansion.
 *
 * class_*_is_conditional lets scoped_guard() prove to the compiler that an
 * unconditional guard body is never skipped.
 */
#define __DEFINE_CLASS_IS_CONDITIONAL(_name, _is_cond)	\
static __maybe_unused const bool class_##_name##_is_conditional = _is_cond

#define __GUARD_IS_ERR(_ptr)                                       \
	({                                                         \
		unsigned long _rc = (__force unsigned long)(_ptr); \
		unlikely((_rc - 1) >= -MAX_ERRNO - 1);             \
	})

#define __DEFINE_GUARD_LOCK_PTR(_name, _exp)                                \
	static __always_inline void *class_##_name##_lock_ptr(class_##_name##_t *_T) \
	{                                                                   \
		void *_ptr = (void *)(__force unsigned long)*(_exp);        \
		if (IS_ERR(_ptr)) {                                         \
			_ptr = NULL;                                        \
		}                                                           \
		return _ptr;                                                \
	}

#define DEFINE_CLASS_IS_GUARD(_name)		\
	__DEFINE_CLASS_IS_CONDITIONAL(_name, false);	\
	__DEFINE_GUARD_LOCK_PTR(_name, _T)

#define DEFINE_GUARD(_name, _type, _lock, _unlock) \
	DEFINE_CLASS(_name, _type, if (!__GUARD_IS_ERR(_T)) { _unlock; }, ({ _lock; _T; }), _type _T); \
	DEFINE_CLASS_IS_GUARD(_name)

#define guard(_name) \
	CLASS(_name, __UNIQUE_ID(guard))

#define __guard_ptr(_name) class_##_name##_lock_ptr
#define __is_cond_ptr(_name) class_##_name##_is_conditional

/*
 * Helper macro for scoped_guard().
 *
 * Note that the "!__is_cond_ptr(_name)" part of the condition ensures that
 * compiler would be sure that for the unconditional locks the body of the
 * loop (caller-provided code glued to the else clause) could not be skipped.
 * It is needed because the other part - "__guard_ptr(_name)(&scope)" - is too
 * hard to deduce (even if could be proven true for unconditional locks).
 */
#define __scoped_guard(_name, _label, args...)				\
	for (CLASS(_name, scope)(args);					\
	     __guard_ptr(_name)(&scope) || !__is_cond_ptr(_name);	\
	     ({ goto _label; }))					\
		if (0) {						\
_label:									\
			break;						\
		} else

#define scoped_guard(_name, args...)	\
	__scoped_guard(_name, __UNIQUE_ID(label), args)

/*
 * DEFINE_LOCK_GUARD_1(name, type, lock, unlock, ...):
 *
 * Like DEFINE_GUARD(), but the guard object is a struct carrying the lock
 * pointer plus whatever extra members __VA_ARGS__ declares (a flags word for
 * the irqsave variants, typically).  Inside @lock / @unlock, _T points at
 * that struct, so _T->lock is the lock and _T->flags the extra state.
 */
#define __DEFINE_UNLOCK_GUARD(_name, _type, _unlock, ...)		\
typedef struct {							\
	_type *lock;							\
	__VA_ARGS__;							\
} class_##_name##_t;							\
									\
static __always_inline void class_##_name##_destructor(class_##_name##_t *_T) \
{									\
	if (!__GUARD_IS_ERR(_T->lock)) { _unlock; }			\
}									\
									\
__DEFINE_GUARD_LOCK_PTR(_name, &_T->lock)

#define __DEFINE_LOCK_GUARD_1(_name, _type, _lock)			\
static __always_inline class_##_name##_t class_##_name##_constructor(_type *l) \
{									\
	class_##_name##_t _t = { .lock = l }, *_T = &_t;		\
	_lock;								\
	return _t;							\
}

#define DEFINE_LOCK_GUARD_1(_name, _type, _lock, _unlock, ...)		\
__DEFINE_CLASS_IS_CONDITIONAL(_name, false);				\
__DEFINE_UNLOCK_GUARD(_name, _type, _unlock, __VA_ARGS__)		\
__DEFINE_LOCK_GUARD_1(_name, _type, _lock)

#endif /* __LINUX_GUARDS_H */
