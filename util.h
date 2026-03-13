#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define znew(sample) ((typeof(sample) *)xzalloc(sizeof(sample)))
#define znew_n(sample, n) ((typeof(sample) *)xzalloc_n(n, sizeof(sample)))

static inline void *
xzalloc_n(int n, size_t size)
{
	void *ptr = calloc(n, size);
	if (!ptr) {
		fprintf(stderr, "calloc failed\n");
		abort();
	}
	return ptr;
}

static inline void *
xzalloc(size_t size)
{
	return xzalloc_n(1, size);
}
