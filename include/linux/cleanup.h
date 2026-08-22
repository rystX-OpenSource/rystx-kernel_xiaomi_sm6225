/* SPDX-License-Identifier: GPL-2.0 */
#ifndef __LINUX_GUARDS_H
#define __LINUX_GUARDS_H

#include <linux/args.h>
#include <linux/compiler.h>
#include <linux/err.h>

/*
 * Backport of the mainline scope-based cleanup helpers, reduced to the
 * class/guard subset.  The __free()/no_free_ptr() pointer-ownership half of
 * upstream cleanup.h is deliberately omitted: nothing in this tree uses it,
 * and it drags in further mainline plumbing (__get_and_null(),
 * __must_check_fn()) that has no 4.19 counterpart.
 *
 * DEFINE_CLASS(name, type, exit, init, init_args...):
 *	declares the CLASS(name, var) initializer and the destructor that
 *	runs on scope exit.
 *
 * DEFINE_GUARD(name, type, lock, unlock):
 *	a DEFINE_CLASS() whose type is the guarded object itself, for use
 *	with guard(name)(obj) and scoped_guard(name, obj).
 *
 * DEFINE_LOCK_GUARD_1(name, type, lock, unlock, ...):
 *	as DEFINE_GUARD() but the class carries extra state alongside the
 *	lock pointer, reachable as _T->member from the lock/unlock
 *	statements.  This is what the MuQSS rq guards are built on.
 */

/*
 * DEFINE_CLASS(name, type, exit, init, init_args...):
 *	simple wrapper around __attribute__((cleanup(func))).
 *
 * CLASS(name, var)(args...):
 *	declares variable @var as an instance of the named class
 */

#define DEFINE_CLASS(_name, _type, _exit, _init, _init_args...)		\
typedef _type class_##_name##_t;					\
static __always_inline void class_##_name##_destructor(_type *p)	\
{ _type _T = *p; _exit; }						\
static __always_inline _type class_##_name##_constructor(_init_args)	\
{ _type t = _init; return t; }

#define EXTEND_CLASS(_name, ext, _init, _init_args...)			\
typedef class_##_name##_t class_##_name##ext##_t;			\
static __always_inline void class_##_name##ext##_destructor(class_##_name##_t *p) \
{ class_##_name##_destructor(p); }					\
static __always_inline class_##_name##_t class_##_name##ext##_constructor(_init_args) \
{ class_##_name##_t t = _init; return t; }

#define CLASS(_name, var)						\
	class_##_name##_t var __cleanup(class_##_name##_destructor) =	\
		class_##_name##_constructor

/*
 * DEFINE_GUARD(name, type, lock, unlock):
 *	trivial wrapper around DEFINE_CLASS() above specifically
 *	for locks.
 *
 * DEFINE_GUARD_COND(name, ext, condlock)
 *	wrapper around EXTEND_CLASS above to add conditional lock
 *	variants to a base class, eg. mutex_trylock() or
 *	mutex_lock_interruptible().
 *
 * guard(name):
 *	an anonymous instance of the (guard) class, not recommended for
 *	conditional locks.
 *
 * scoped_guard (name, args...) { }:
 *	similar to CLASS(name, scope)(args), except the variable (with the
 *	explicit name 'scope') is declard in a for-loop such that its scope is
 *	bound to the next (compound) statement.
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
	}                                                                   \
	static __always_inline int class_##_name##_lock_err(class_##_name##_t *_T) \
	{                                                                   \
		long _rc = (__force unsigned long)*(_exp);                  \
		if (!_rc) {                                                 \
			_rc = -EBUSY;                                       \
		}                                                           \
		if (!IS_ERR_VALUE(_rc)) {                                   \
			_rc = 0;                                            \
		}                                                           \
		return _rc;                                                 \
	}

#define DEFINE_CLASS_IS_GUARD(_name) \
	__DEFINE_CLASS_IS_CONDITIONAL(_name, false); \
	__DEFINE_GUARD_LOCK_PTR(_name, _T)

#define DEFINE_CLASS_IS_COND_GUARD(_name) \
	__DEFINE_CLASS_IS_CONDITIONAL(_name, true); \
	__DEFINE_GUARD_LOCK_PTR(_name, _T)

#define DEFINE_GUARD(_name, _type, _lock, _unlock) \
	DEFINE_CLASS(_name, _type, if (!__GUARD_IS_ERR(_T)) { _unlock; }, ({ _lock; _T; }), _type _T); \
	DEFINE_CLASS_IS_GUARD(_name)

#define DEFINE_GUARD_COND_4(_name, _ext, _lock, _cond)			\
	__DEFINE_CLASS_IS_CONDITIONAL(_name##_ext, true);		\
	EXTEND_CLASS(_name, _ext,					\
		     ({ void *_t = _T;					\
		        int _RET = (_lock);				\
		        if (_T && !(_cond)) _t = ERR_PTR(_RET);		\
		        _t; }),						\
		     class_##_name##_t _T)				\
	static __always_inline void *class_##_name##_ext##_lock_ptr(class_##_name##_t *_T) \
	{ return class_##_name##_lock_ptr(_T); }			\
	static __always_inline int class_##_name##_ext##_lock_err(class_##_name##_t *_T) \
	{ return class_##_name##_lock_err(_T); }

#define DEFINE_GUARD_COND_3(_name, _ext, _lock) \
	DEFINE_GUARD_COND_4(_name, _ext, _lock, _RET)

#define DEFINE_GUARD_COND(X...) CONCATENATE(DEFINE_GUARD_COND_, COUNT_ARGS(X))(X)

#define guard(_name) \
	CLASS(_name, __UNIQUE_ID(guard))

#define __guard_ptr(_name) class_##_name##_lock_ptr
#define __guard_err(_name) class_##_name##_lock_err
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
 * Additional helper macros for generating lock guards with types, either for
 * locks that don't have a native type (eg. RCU, preempt) or those that need a
 * 'fat' pointer (eg. spin_lock_irqsave).
 *
 * DEFINE_LOCK_GUARD_0(name, lock, unlock, ...)
 * DEFINE_LOCK_GUARD_1(name, type, lock, unlock, ...)
 * DEFINE_LOCK_GUARD_1_COND(name, ext, condlock)
 *
 * will result in the following type:
 *
 *   typedef struct {
 *	type *lock;		// 'type := void' for the _0 variant
 *	__VA_ARGS__;
 *   } class_##name##_t;
 *
 * As above, both _lock and _unlock are statements, except this time '_T' will
 * be a pointer to the above struct.
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

#define __DEFINE_LOCK_GUARD_0(_name, _lock)				\
static __always_inline class_##_name##_t class_##_name##_constructor(void) \
{									\
	class_##_name##_t _t = { .lock = (void*)1 },			\
			 *_T __maybe_unused = &_t;			\
	_lock;								\
	return _t;							\
}

#define DEFINE_LOCK_GUARD_1(_name, _type, _lock, _unlock, ...)		\
__DEFINE_CLASS_IS_CONDITIONAL(_name, false);				\
__DEFINE_UNLOCK_GUARD(_name, _type, _unlock, __VA_ARGS__)		\
__DEFINE_LOCK_GUARD_1(_name, _type, _lock)

#define DEFINE_LOCK_GUARD_0(_name, _lock, _unlock, ...)			\
__DEFINE_CLASS_IS_CONDITIONAL(_name, false);				\
__DEFINE_UNLOCK_GUARD(_name, void, _unlock, __VA_ARGS__)		\
__DEFINE_LOCK_GUARD_0(_name, _lock)

#define DEFINE_LOCK_GUARD_1_COND_4(_name, _ext, _lock, _cond)		\
	__DEFINE_CLASS_IS_CONDITIONAL(_name##_ext, true);		\
	EXTEND_CLASS(_name, _ext,					\
		     ({ class_##_name##_t _t = { .lock = l }, *_T = &_t;\
		        int _RET = (_lock);                             \
		        if (_T->lock && !(_cond)) _T->lock = ERR_PTR(_RET);\
			_t; }),						\
		     typeof(((class_##_name##_t *)0)->lock) l)		\
	static __always_inline void * class_##_name##_ext##_lock_ptr(class_##_name##_t *_T) \
	{ return class_##_name##_lock_ptr(_T); } \
	static __always_inline int class_##_name##_ext##_lock_err(class_##_name##_t *_T) \
	{ return class_##_name##_lock_err(_T); }

#define DEFINE_LOCK_GUARD_1_COND_3(_name, _ext, _lock) \
	DEFINE_LOCK_GUARD_1_COND_4(_name, _ext, _lock, _RET)

#define DEFINE_LOCK_GUARD_1_COND(X...) CONCATENATE(DEFINE_LOCK_GUARD_1_COND_, COUNT_ARGS(X))(X)

#endif /* __LINUX_GUARDS_H */
