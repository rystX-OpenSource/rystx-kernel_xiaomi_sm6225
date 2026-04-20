#ifndef __KSU_H_KSUD_ESCAPE
#define __KSU_H_KSUD_ESCAPE

#if LINUX_VERSION_CODE < KERNEL_VERSION(4, 14, 0) && LINUX_VERSION_CODE >= KERNEL_VERSION(4, 2, 0) && !defined(CONFIG_KRETPROBES)
static noinline void sys_execve_escape_ksud_internal(void *filename);
static noinline void kernel_execve_escape_ksud_internal(void *filename);

#ifdef CONFIG_JUMP_LABEL
DEFINE_STATIC_KEY_TRUE(ksu_boot_incomplete_key);
static inline void sys_execve_escape_ksud(void *filename)
{
	if (static_branch_likely(&ksu_boot_incomplete_key))
		sys_execve_escape_ksud_internal(filename);
}
static inline void kernel_execve_escape_ksud(void *filename)
{
	if (static_branch_likely(&ksu_boot_incomplete_key))
		kernel_execve_escape_ksud_internal(filename);
}
#else
// the jump_label at home:
// basically test + jz vs call + ret
// likely worse on newer chips, but newer chips have jump_label so
__attribute__((hot)) static void ksud_exec_escape_noop(void *filename) { } // no-op

static void (*kernel_execve_escape_ksud)(void *filename) __read_mostly = kernel_execve_escape_ksud_internal;
static void (*sys_execve_escape_ksud)(void *filename) __read_mostly = sys_execve_escape_ksud_internal;

#endif // CONFIG_JUMP_LABEL

#else
static inline void sys_execve_escape_ksud(void *filename) { } // no-op
static inline void kernel_execve_escape_ksud(void *filename) { } // no-op
#endif // < 4.14 && >= 4.2 && !KRETPROBES

static void ksud_escape_init();
static void ksud_escape_exit();

#endif // __KSU_H_KSUD_ESCAPE
