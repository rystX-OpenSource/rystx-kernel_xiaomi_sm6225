// SPDX-License-Identifier: GPL-2.0
/*
 * taglmk_bench - the page corpus.
 *
 * A compressor's throughput and its ratio both depend far more on what it is
 * fed than on anything the kernel does, so two runs are only comparable if they
 * were fed the same bytes.  That is the whole design here: the corpus is a pure
 * function of a seed and a page index.  Page 900 of a run is byte for byte page
 * 900 of any other run with the same seed, on any device, whatever order the
 * pages were asked for in and however many of them were asked for.  The seed
 * and the mixture are written into the saved file, and a comparison refuses to
 * score two runs that disagree about either.
 *
 * Five shapes of page, in fixed proportions, because a single shape measures
 * the wrong thing.  Feed a compressor nothing but random bytes and every page
 * turns incompressible, the recompression pass has nothing to select, and
 * ZRAM-IR looks like pure overhead; feed it nothing but zeroes and zram
 * deduplicates the lot and never compresses anything at all.  Both are real
 * behaviours worth exercising and neither is a phone.  The default mixture is
 * roughly what an Android anonymous working set looks like: mostly heap and
 * text with a scattering of zero pages and a minority that will not compress.
 *
 * Copyright (C) 2026 iDeadXS <datarafi43@gmail.com>
 */
#define _GNU_SOURCE

#include <stdio.h>
#include <string.h>

#include "bench.h"

/*
 * Parts per hundred.  These are fixed rather than tunable on purpose: a mixture
 * the operator can change is a mixture two changelog entries will disagree
 * about, and the point of the corpus is that they cannot.
 */
static const unsigned int bench_default_weight[BENCH_PAGE_KINDS] = {
	[BENCH_PAGE_ZERO]	= 5,
	[BENCH_PAGE_TEXT]	= 30,
	[BENCH_PAGE_HEAP]	= 35,
	[BENCH_PAGE_MIXED]	= 20,
	[BENCH_PAGE_RANDOM]	= 10,
};

static const char bench_kind_tag[BENCH_PAGE_KINDS] = {
	[BENCH_PAGE_ZERO]	= 'z',
	[BENCH_PAGE_TEXT]	= 't',
	[BENCH_PAGE_HEAP]	= 'h',
	[BENCH_PAGE_MIXED]	= 'm',
	[BENCH_PAGE_RANDOM]	= 'r',
};

/* splitmix64 again, and for the same reason: reproducible everywhere. */
static uint64_t bench_mix(uint64_t *state)
{
	uint64_t z = (*state += 0x9e3779b97f4a7c15ull);

	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;

	return z ^ (z >> 31);
}

/*
 * The generator for one page is seeded from the corpus seed and the page index
 * together, which is what makes a page depend on nothing but its own address in
 * the sequence.  Mixing the two through splitmix64's finaliser rather than
 * adding them keeps adjacent pages from being near relatives.
 */
static uint64_t bench_page_seed(const struct bench_corpus *c, uint64_t index)
{
	uint64_t state = c->seed;

	state ^= index * 0xd1342543de82ef95ull;

	return bench_mix(&state);
}

void bench_corpus_init(struct bench_corpus *c, size_t page_size,
		       uint64_t nr_pages, uint32_t seed)
{
	memset(c, 0, sizeof(*c));
	c->page_size = page_size;
	c->nr_pages = nr_pages;
	c->seed = seed;
	memcpy(c->weight, bench_default_weight, sizeof(c->weight));
}

/*
 * Which shape page @index is.  Derived from the page's own seed, so the mixture
 * holds over any long enough run of pages without the sequence having to be
 * generated in order or all at once.
 */
static enum bench_page_kind bench_kind_of(const struct bench_corpus *c,
					  uint64_t seed)
{
	unsigned int total = 0, pick;

	for (unsigned int i = 0; i < BENCH_PAGE_KINDS; i++)
		total += c->weight[i];

	if (!total)
		return BENCH_PAGE_HEAP;

	pick = (unsigned int)(seed % total);

	for (unsigned int i = 0; i < BENCH_PAGE_KINDS; i++) {
		if (pick < c->weight[i])
			return (enum bench_page_kind)i;
		pick -= c->weight[i];
	}

	return BENCH_PAGE_HEAP;
}

/* A word list, so the text pages have a vocabulary rather than an alphabet. */
static const char * const bench_words[] = {
	"the", "page", "kernel", "memory", "task", "reclaim", "swap", "zram",
	"compress", "anon", "file", "cache", "budget", "pressure", "level",
	"window", "sample", "factor", "burst", "gain", "slot", "huge", "idle",
	"scan", "sweep", "priority", "algorithm", "stream", "device", "block",
	"and", "of", "to", "in", "for", "with", "that", "this", "from", "into",
};

#define BENCH_NR_WORDS	(sizeof(bench_words) / sizeof(bench_words[0]))

/*
 * Prose-like: words from a small vocabulary, separated by spaces, with a line
 * break now and then.  This is the shape lz4 and zstd both do well on, and it
 * is what a page of a dex string pool or a log buffer looks like.
 */
static void bench_fill_text(unsigned char *out, size_t len, uint64_t *state)
{
	size_t at = 0;

	while (at < len) {
		const char *w = bench_words[bench_mix(state) % BENCH_NR_WORDS];
		size_t n = strlen(w);

		if (at + n + 1 > len)
			break;

		memcpy(out + at, w, n);
		at += n;
		out[at++] = (bench_mix(state) % 12) ? ' ' : '\n';
	}

	/* Whatever is left is padded, not left uninitialised. */
	if (at < len)
		memset(out + at, ' ', len - at);
}

/*
 * Heap-like: eight byte records, most of them a pointer into a plausible
 * address range, some a small integer, and long runs where a record repeats.
 * Pointers share their high bytes, which is exactly the redundancy a compressor
 * finds in a real heap and the reason heap pages compress at all.
 */
static void bench_fill_heap(unsigned char *out, size_t len, uint64_t *state)
{
	const uint64_t base = 0x0000007f00000000ull;
	uint64_t word = base;
	size_t at = 0;

	while (at + sizeof(uint64_t) <= len) {
		uint64_t r = bench_mix(state);
		unsigned int run = 1 + (unsigned int)(r % 6);

		switch ((r >> 8) % 4) {
		case 0:
			word = base | (bench_mix(state) & 0xffffffffull);
			break;
		case 1:
			word = bench_mix(state) % 4096;
			break;
		case 2:
			/* A near neighbour of the last one: a linked list. */
			word += 0x40;
			break;
		default:
			/* Repeat, which is what an array of the same tag is. */
			break;
		}

		while (run-- && at + sizeof(uint64_t) <= len) {
			memcpy(out + at, &word, sizeof(word));
			at += sizeof(word);
		}
	}

	if (at < len)
		memset(out + at, 0, len - at);
}

static void bench_fill_random(unsigned char *out, size_t len, uint64_t *state)
{
	size_t at = 0;

	while (at + sizeof(uint64_t) <= len) {
		uint64_t r = bench_mix(state);

		memcpy(out + at, &r, sizeof(r));
		at += sizeof(r);
	}

	while (at < len)
		out[at++] = (unsigned char)bench_mix(state);
}

/*
 * Structure with incompressible islands: a heap page with a few random blocks
 * dropped into it.  This is the shape that decides whether a page ends up over
 * huge_class_size, and therefore the shape the recompression pass lives or dies
 * on, so it is worth having in the mixture in its own right.
 */
static void bench_fill_mixed(unsigned char *out, size_t len, uint64_t *state)
{
	unsigned int islands;

	bench_fill_heap(out, len, state);

	islands = 1 + (unsigned int)(bench_mix(state) % 4);

	while (islands--) {
		size_t block = len / 8;
		size_t at;

		if (!block || block > len)
			break;

		at = (size_t)(bench_mix(state) % (len - block + 1));
		bench_fill_random(out + at, block, state);
	}
}

void bench_corpus_page(const struct bench_corpus *c, uint64_t index,
		       unsigned char *out)
{
	uint64_t state = bench_page_seed(c, index);
	enum bench_page_kind kind = bench_kind_of(c, state);

	switch (kind) {
	case BENCH_PAGE_ZERO:
		memset(out, 0, c->page_size);
		break;
	case BENCH_PAGE_TEXT:
		bench_fill_text(out, c->page_size, &state);
		break;
	case BENCH_PAGE_HEAP:
		bench_fill_heap(out, c->page_size, &state);
		break;
	case BENCH_PAGE_MIXED:
		bench_fill_mixed(out, c->page_size, &state);
		break;
	case BENCH_PAGE_RANDOM:
		bench_fill_random(out, c->page_size, &state);
		break;
	case BENCH_PAGE_KINDS:
	default:
		/*
		 * Unreachable, but a page that is never written is a page whose
		 * verification compares uninitialised memory, so fill it.
		 */
		memset(out, 0, c->page_size);
		break;
	}
}

void bench_corpus_describe(const struct bench_corpus *c, char *buf, size_t len)
{
	size_t at = 0;

	if (!len)
		return;

	buf[0] = '\0';

	for (unsigned int i = 0; i < BENCH_PAGE_KINDS; i++) {
		int n = snprintf(buf + at, len - at, "%s%c=%u",
				 at ? "," : "",
				 bench_kind_tag[i], c->weight[i]);

		if (n < 0 || (size_t)n >= len - at)
			return;

		at += (size_t)n;
	}
}
