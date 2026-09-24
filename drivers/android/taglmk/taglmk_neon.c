// SPDX-License-Identifier: GPL-2.0
/*
 * TAGLMK - ARM64 NEON prediction accelerator.
 *
 * Eight-wide Q4.4 window reductions used by taglmk_predict.c.  Every s16 lane
 * is widened to s32 before it is accumulated, so the result is bit-identical
 * to the scalar intfp-32 reference in taglmk_predict.c; NEON only makes it
 * faster.  All vector work is bracketed by kernel_neon_begin()/end() and is
 * only ever reached after taglmk_neon_usable() has confirmed SIMD is allowed
 * in the current (kthread, process) context.
 */
#include <asm/neon.h>
#include <asm/simd.h>
#include <asm/neon-intrinsics.h>

#include "taglmk.h"

bool taglmk_neon_usable(void)
{
	return cpu_has_neon() && may_use_simd();
}

s32 taglmk_neon_sum_q44(const q4_4_t *s, int n)
{
	s32 out, tail = 0;
	int i = 0;

	kernel_neon_begin();
	{
		int32x4_t acc = vdupq_n_s32(0);

		for (; i + 8 <= n; i += 8) {
			int16x8_t v = vld1q_s16(&s[i]);

			acc = vaddq_s32(acc, vaddl_s16(vget_low_s16(v),
						       vget_high_s16(v)));
		}
		out = vaddvq_s32(acc);
	}
	kernel_neon_end();

	for (; i < n; i++)
		tail += s[i];
	return out + tail;
}

s32 taglmk_neon_absdev_q44(const q4_4_t *s, int n, q4_4_t mean)
{
	s32 out, tail = 0;
	int i = 0;

	kernel_neon_begin();
	{
		int16x8_t vmean = vdupq_n_s16(mean);
		int32x4_t acc = vdupq_n_s32(0);

		for (; i + 8 <= n; i += 8) {
			int16x8_t v = vld1q_s16(&s[i]);
			int16x8_t d = vabsq_s16(vsubq_s16(v, vmean));

			acc = vaddq_s32(acc, vaddl_s16(vget_low_s16(d),
						       vget_high_s16(d)));
		}
		out = vaddvq_s32(acc);
	}
	kernel_neon_end();

	for (; i < n; i++) {
		s32 d = (s32)s[i] - mean;

		tail += (d < 0) ? -d : d;
	}
	return out + tail;
}
