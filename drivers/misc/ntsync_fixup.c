// SPDX-License-Identifier: GPL-2.0-only
/*
 * ntsync_fixup.c - /dev/ntsync 的 SELinux 上下文与权限修复
 *
 * 从 ntsync.c 外移的独立 fixup：misc 设备已在 ntsync_init 中注册，
 * /dev/ntsync 立即可见，此处延后 2s 执行仅作保险（Q3 语义不变）。
 * 已适配 4.19 API：__vfs_setxattr_noperm 在该版本不带 idmap/userns
 * 参数，因此去掉 linux/mnt_idmapping.h 与 nop_mnt_idmap 的使用。
 *
 * 与 ntsync.c 的 cmdline 开关保持一致：复用其全局 ntsync_enabled
 * （由 __setup("ntsync.enabled=") 在 initcall 之前解析）。当驱动被
 * 关闭（ntsync_enabled < 1）时不排入延后 work，避免无谓调度；此前
 * 仅靠 worker 里 kern_path 失败静默返回来间接规避。
 *
 * 注意：ntsync_enabled 与 __vfs_setxattr_noperm 在 4.19 均未导出，
 * 仅在内建（CONFIG_NTSYNC=y）时可链接；改成 =m 会在 modpost 报未
 * 定义符号。
 *
 * iDead@rystX-OpenSource: adapted to 4.19 APIs
 */

#include <linux/fs.h>
#include <linux/init.h>
#include <linux/jiffies.h>
#include <linux/kernel.h>
#include <linux/namei.h>
#include <linux/path.h>
#include <linux/string.h>
#include <linux/workqueue.h>
#include <linux/xattr.h>

/* 定义于 ntsync.c；由 ntsync.enabled= 命令行开关设置 */
extern unsigned int ntsync_enabled;

static struct delayed_work ntsync_perm_work;

static void ntsync_fix_perms_worker(struct work_struct *work)
{
	struct path path;
	const char *ctx = "u:object_r:gpu_device:s0";

	if (!kern_path("/dev/ntsync", LOOKUP_FOLLOW, &path)) {
		struct inode *inode = d_backing_inode(path.dentry);

		if (inode) {
			/* 4.19: __vfs_setxattr_noperm(dentry, name, value, size, flags) */
			__vfs_setxattr_noperm(path.dentry,
					      "security.selinux", ctx, strlen(ctx) + 1, 0);
			inode->i_mode = (inode->i_mode & ~S_IALLUGO) | 0666;
			pr_info("ntsync: Applied 0666 and gpu_device context\n");
		}
		path_put(&path);
	}
	/* kern_path 失败时静默返回，与原实现行为一致 */
}

static int __init ntsync_fixup_init(void)
{
	/* 驱动被 cmdline 关闭时不注册，/dev/ntsync 不存在，无需 fixup */
	if (ntsync_enabled < 1) {
		pr_info("ntsync: fixup skipped (driver disabled via cmdline)\n");
		return 0;
	}

	INIT_DELAYED_WORK(&ntsync_perm_work, ntsync_fix_perms_worker);
	schedule_delayed_work(&ntsync_perm_work, msecs_to_jiffies(2000));

	return 0;
}
late_initcall(ntsync_fixup_init);