# SPDX-License-Identifier: GPL-2.0-only
#
# Userspace unit tests for the FLD decision math (fld_math.h).

CC	?= cc
CFLAGS	?= -O2 -Wall -Wextra

test: test_fld_math
	./test_fld_math

test_fld_math: test_fld_math.c ../fld_math.h
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f test_fld_math

.PHONY: test clean
