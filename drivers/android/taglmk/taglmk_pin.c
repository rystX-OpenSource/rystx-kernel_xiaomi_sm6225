// SPDX-License-Identifier: GPL-2.0
/*
 * TAGLMK - package pinning.
 *
 * A small user-managed set of Android package names.  A task whose command
 * line matches a pinned package (either exactly, or as the "pkg" part of a
 * "pkg:subprocess" name) is classified PINNED and given materially longer
 * survivability by the policy engine.
 *
 * Managed through /sys/kernel/taglmk/packages:
 *   read           - one pinned package per line
 *   write "pkg"    - pin a package
 *   write "-pkg"   - unpin a package
 *   write ""       - (empty) unpin everything
 */
#include <linux/ctype.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/string.h>
#include <linux/sysfs.h>

#include "taglmk.h"

#define TAGLMK_PKG_MAX		128	/* max package-name length */
#define TAGLMK_PIN_MAX		64	/* max pinned packages */
#define TAGLMK_CMDLINE_MAX	256

struct taglmk_pin_set {
	char		names[TAGLMK_PIN_MAX][TAGLMK_PKG_MAX];
	unsigned int	count;
	spinlock_t	lock;
};

static struct taglmk_pin_set pin_set = {
	.lock = __SPIN_LOCK_UNLOCKED(pin_set.lock),
};

/* Length of the base package name (up to a ':' subprocess separator). */
static size_t base_pkg_len(const char *s)
{
	const char *c = strchr(s, ':');

	return c ? (size_t)(c - s) : strlen(s);
}

static int pin_add(const char *name)
{
	unsigned int i;
	int ret = 0;

	if (!*name || strlen(name) >= TAGLMK_PKG_MAX)
		return -EINVAL;

	spin_lock(&pin_set.lock);
	for (i = 0; i < pin_set.count; i++) {
		if (!strcmp(pin_set.names[i], name))
			goto out;	/* already pinned */
	}
	if (pin_set.count >= TAGLMK_PIN_MAX) {
		ret = -ENOSPC;
		goto out;
	}
	strscpy(pin_set.names[pin_set.count], name, TAGLMK_PKG_MAX);
	pin_set.count++;
out:
	spin_unlock(&pin_set.lock);
	return ret;
}

static void pin_remove(const char *name)
{
	unsigned int i;

	spin_lock(&pin_set.lock);
	for (i = 0; i < pin_set.count; i++) {
		if (strcmp(pin_set.names[i], name))
			continue;
		/* compact the tail down over the removed slot */
		if (i != pin_set.count - 1)
			memmove(pin_set.names[i], pin_set.names[i + 1],
				(pin_set.count - i - 1) * TAGLMK_PKG_MAX);
		pin_set.count--;
		break;
	}
	spin_unlock(&pin_set.lock);
}

bool taglmk_is_pinned(struct task_struct *tsk)
{
	char cmd[TAGLMK_CMDLINE_MAX];
	int res, len;
	size_t base;
	unsigned int i;
	bool match = false;

	/* Racy fast path: nothing pinned, nothing to match. */
	if (!READ_ONCE(pin_set.count))
		return false;
	/* Kernel threads have no command line and can never be an app. */
	if (!tsk->mm)
		return false;

	res = get_cmdline(tsk, cmd, sizeof(cmd) - 1);
	if (res <= 0)
		return false;
	len = min(res, (int)sizeof(cmd) - 1);
	cmd[len] = '\0';
	/* argv[0] is the Android process name; stop at its terminating NUL. */
	len = strnlen(cmd, len);
	cmd[len] = '\0';
	if (!len)
		return false;

	base = base_pkg_len(cmd);

	spin_lock(&pin_set.lock);
	for (i = 0; i < pin_set.count; i++) {
		const char *p = pin_set.names[i];

		if (!strcmp(p, cmd) ||
		    (strlen(p) == base && !strncmp(p, cmd, base))) {
			match = true;
			break;
		}
	}
	spin_unlock(&pin_set.lock);
	return match;
}

/* ------------------------------------------------------------------ */
/* sysfs: /sys/kernel/taglmk/packages */

static ssize_t packages_show(struct kobject *kobj, struct kobj_attribute *attr,
			     char *buf)
{
	unsigned int i;
	int len = 0;

	spin_lock(&pin_set.lock);
	for (i = 0; i < pin_set.count && len < PAGE_SIZE - TAGLMK_PKG_MAX; i++)
		len += scnprintf(buf + len, PAGE_SIZE - len, "%s\n",
				 pin_set.names[i]);
	spin_unlock(&pin_set.lock);
	return len;
}

static ssize_t packages_store(struct kobject *kobj, struct kobj_attribute *attr,
			      const char *buf, size_t count)
{
	char tok[TAGLMK_PKG_MAX];
	const char *p = buf;
	size_t n;

	/* Take the first whitespace-delimited token. */
	while (*p && isspace(*p))
		p++;
	n = 0;
	while (p[n] && !isspace(p[n]))
		n++;

	if (n == 0) {			/* empty write clears the whole set */
		spin_lock(&pin_set.lock);
		pin_set.count = 0;
		spin_unlock(&pin_set.lock);
		return count;
	}
	if (n >= sizeof(tok))
		return -EINVAL;

	memcpy(tok, p, n);
	tok[n] = '\0';

	if (tok[0] == '-')
		pin_remove(tok + 1);
	else
		pin_add(tok);

	return count;
}

static struct kobj_attribute packages_attr =
	__ATTR(packages, 0644, packages_show, packages_store);

int taglmk_pin_init(struct kobject *parent)
{
	return sysfs_create_file(parent, &packages_attr.attr);
}

void taglmk_pin_exit(void)
{
	/* The kobject is torn down by the core; nothing else to release. */
}
