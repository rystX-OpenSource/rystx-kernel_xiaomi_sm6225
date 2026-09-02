#!/bin/sh
# SPDX-License-Identifier: GPL-2.0
#
# Build taglmk_bench with the Android NDK.
#
#   ANDROID_NDK=~/Android/Sdk/ndk/26.1.10909125 ./build.sh
#
# Two rules decide how this compiles, and both matter to whether the numbers
# mean anything.
#
# One translation unit per kernel generation, and no link time optimisation.
# kernels_v0.c and kernels_v1.c hold the two NEON generations being compared.
# If the compiler is allowed to see both at once it may inline them into the
# same caller, unify what they have in common, and hoist the shared setup out of
# the timing loop, and the difference measured is then a property of the
# optimiser rather than of the code the kernel actually runs.
#
# Plain -O2, no vectoriser flags.  The kernel this benchmark speaks for is built
# at -O2 with neither -ftree-vectorize nor -fno-vectorize, so its scalar C is
# compiled exactly the way the scalar twins here are.  Turning autovectorisation
# up would measure a compiler that never touched the driver.
#
# Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>

set -eu

: "${ANDROID_NDK:=${NDK:-}}"
: "${API:=29}"
: "${ARCH:=aarch64}"
: "${OUT:=taglmk_bench}"

if [ -z "$ANDROID_NDK" ]; then
	echo "set ANDROID_NDK to the NDK root, e.g." >&2
	echo "  ANDROID_NDK=~/Android/Sdk/ndk/26.1.10909125 $0" >&2
	exit 1
fi

case "$(uname -s)" in
Linux)	HOST=linux-x86_64 ;;
Darwin)	HOST=darwin-x86_64 ;;
*)	echo "unknown host $(uname -s)" >&2; exit 1 ;;
esac

BIN="$ANDROID_NDK/toolchains/llvm/prebuilt/$HOST/bin"
CC="$BIN/${ARCH}-linux-android${API}-clang"

if [ ! -x "$CC" ]; then
	echo "no compiler at $CC" >&2
	echo "check ANDROID_NDK, API=$API and ARCH=$ARCH" >&2
	exit 1
fi

SRC="main.c util.c report.c timing.c corpus.c cpu_suite.c zram_suite.c
     kernels_scalar.c kernels_v0.c kernels_v1.c"

CFLAGS="-std=gnu11 -O2 -g -fno-lto
	-Wall -Wextra -Wshadow -Wvla -Wformat=2 -Wcast-align
	-Wstrict-prototypes -Wmissing-prototypes
	-Werror=implicit-function-declaration
	-D_GNU_SOURCE
	-fstack-protector-strong -D_FORTIFY_SOURCE=2"

LDFLAGS="-fno-lto -static"

# -static so the binary runs from /data/local/tmp on any image, whatever its
# libc happens to be.  Drop it if the resulting size is a problem.

rm -rf build
mkdir -p build

for f in $SRC; do
	echo "  CC  $f"
	# shellcheck disable=SC2086
	"$CC" $CFLAGS -c "$f" -o "build/${f%.c}.o"
done

echo "  LD  $OUT"
# shellcheck disable=SC2086
"$CC" $LDFLAGS build/*.o -lm -o "$OUT"

echo
echo "built $OUT"
"$BIN/llvm-size" "$OUT" 2>/dev/null || true

cat <<'USAGE'

push it and run it:
  adb push taglmk_bench /data/local/tmp/
  adb shell su -c 'cd /data/local/tmp && ./taglmk_bench -v -o before.txt'
  ... flash the new kernel ...
  adb shell su -c 'cd /data/local/tmp && ./taglmk_bench -v -o after.txt \
      -b before.txt'
USAGE
