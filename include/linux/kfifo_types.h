/* SPDX-License-Identifier: GPL-2.0-or-later */
/*
 * A generic kernel FIFO implementation -- type definitions only.
 *
 * Copyright (C) 2013 Stefani Seibold <stefani@seibold.net>
 */

/*
 * Marie LRU's kcompressd queue lives inside struct pglist_data, so
 * <linux/mmzone.h> needs the complete definition of struct kfifo.  It cannot
 * get it from <linux/kfifo.h>, because this tree's kfifo.h pulls in
 * <linux/scatterlist.h> for the DMA helpers, and scatterlist.h pulls in
 * <linux/mm.h> -- which includes mmzone.h.  Including kfifo.h from mmzone.h
 * would therefore close an include cycle and leave scatterlist.h compiling
 * against a half-emitted mm.h.
 *
 * Linux 6.19.8 does not have this problem: its kfifo.h only forward-declares
 * struct scatterlist (include/linux/kfifo.h:47) and so is free of any mm
 * dependency.  Rather than drop the scatterlist include from this tree's
 * kfifo.h -- which 99 in-tree files include, an unknown number of them
 * relying on the transitive <linux/mm.h> -- the mm-free part of kfifo.h is
 * split out here.  kfifo.h includes this header, so every existing user sees
 * exactly what it saw before; mmzone.h includes only this header, and so
 * never reaches scatterlist.h.
 */

#ifndef _LINUX_KFIFO_TYPES_H
#define _LINUX_KFIFO_TYPES_H

struct __kfifo {
	unsigned int	in;
	unsigned int	out;
	unsigned int	mask;
	unsigned int	esize;
	void		*data;
};

#define __STRUCT_KFIFO_COMMON(datatype, recsize, ptrtype) \
	union { \
		struct __kfifo	kfifo; \
		datatype	*type; \
		const datatype	*const_type; \
		char		(*rectype)[recsize]; \
		ptrtype		*ptr; \
		ptrtype const	*ptr_const; \
	}

#define __STRUCT_KFIFO(type, size, recsize, ptrtype) \
{ \
	__STRUCT_KFIFO_COMMON(type, recsize, ptrtype); \
	type		buf[((size < 2) || (size & (size - 1))) ? -1 : size]; \
}

#define STRUCT_KFIFO(type, size) \
	struct __STRUCT_KFIFO(type, size, 0, type)

#define __STRUCT_KFIFO_PTR(type, recsize, ptrtype) \
{ \
	__STRUCT_KFIFO_COMMON(type, recsize, ptrtype); \
	type		buf[0]; \
}

#define STRUCT_KFIFO_PTR(type) \
	struct __STRUCT_KFIFO_PTR(type, 0, type)

/*
 * define compatibility "struct kfifo" for dynamic allocated fifos
 */
struct kfifo __STRUCT_KFIFO_PTR(unsigned char, 0, void);

#define STRUCT_KFIFO_REC_1(size) \
	struct __STRUCT_KFIFO(unsigned char, size, 1, void)

#define STRUCT_KFIFO_REC_2(size) \
	struct __STRUCT_KFIFO(unsigned char, size, 2, void)

/*
 * define kfifo_rec types
 */
struct kfifo_rec_ptr_1 __STRUCT_KFIFO_PTR(unsigned char, 1, void);
struct kfifo_rec_ptr_2 __STRUCT_KFIFO_PTR(unsigned char, 2, void);

#endif /* _LINUX_KFIFO_TYPES_H */
