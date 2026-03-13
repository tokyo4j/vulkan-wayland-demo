/*
 * Copyright © 2008-2011 Kristian Høgsberg
 * Copyright © 2012 Collabora, Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

#include <assert.h>
#include <stdbool.h>

#include <wayland-server-protocol.h>

// clang-format off
struct vec4f {
	union {
		float el[4];
		struct {
			float x, y, z, w;
		};
	};
};

struct mat4f {
	union {
		struct vec4f col[4];
		float colmaj[4 * 4];
	};
};

#define MAT4F(a00, a01, a02, a03, \
	      a10, a11, a12, a13, \
	      a20, a21, a22, a23, \
	      a30, a31, a32, a33) \
	(struct mat4f){ .colmaj = { \
		a00, a10, a20, a30, \
		a01, a11, a21, a31, \
		a02, a12, a22, a32, \
		a03, a13, a23, a33  \
	}}


static inline void
mat4f_init(struct mat4f *mat)
{
	*mat = MAT4F(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
}

static inline struct mat4f
mat4f_scaling(float x, float y, float z)
{
	return MAT4F(
		x,    0.0f, 0.0f, 0.0f,
		0.0f, y,    0.0f, 0.0f,
		0.0f, 0.0f, z,    0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);
}

static inline struct mat4f
mat4f_translation(float tx, float ty, float tz)
{
	return MAT4F(
		1.0f, 0.0f, 0.0f, tx,
		0.0f, 1.0f, 0.0f, ty,
		0.0f, 0.0f, 1.0f, tz,
		0.0f, 0.0f, 0.0f, 1.0f);
}

static inline struct mat4f
mat4f_rotation_xy(float cos_th, float sin_th)
{
	return MAT4F(
		cos_th, -sin_th, 0.0f, 0.0f,
		sin_th,  cos_th, 0.0f, 0.0f,
		  0.0f,    0.0f, 1.0f, 0.0f,
		  0.0f,    0.0f, 0.0f, 1.0f);
}

static inline struct vec4f
mat4f_mul_vec4f(struct mat4f a, struct vec4f b)
{
	struct vec4f result;
	for (int i = 0; i < 4; i++) {
		result.el[i] =
			a.col[0].el[i] * b.el[0] +
			a.col[1].el[i] * b.el[1] +
			a.col[2].el[i] * b.el[2] +
			a.col[3].el[i] * b.el[3];
	}
	return result;
}

static inline struct mat4f
mat4f_mul_mat4f(struct mat4f a, struct mat4f b)
{
	struct mat4f result;

	for (int i = 0; i < 4; i++) {
		result.col[i] = mat4f_mul_vec4f(a, b.col[i]);
	}

	return result;
}

static inline void
mat4f_scale(struct mat4f *mat, float x, float y, float z)
{
	*mat = mat4f_mul_mat4f(mat4f_scaling(x, y, z), *mat);
}

static inline void
mat4f_translate(struct mat4f *mat, float tx, float ty, float tz)
{
	*mat = mat4f_mul_mat4f(mat4f_translation(tx, ty, tz), *mat);
}

static inline void
mat4f_rotate_xy(struct mat4f *mat, float cos_th, float sin_th)
{
	*mat = mat4f_mul_mat4f(mat4f_rotation_xy(cos_th, sin_th), *mat);
}
