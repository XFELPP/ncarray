/*
 * Copyright (c) 2025-2026 Gabriel Dorlhiac
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "gtest/gtest.h"

#include "ncarray/custom_types.hh"
#include "ncarray/op_traits.hh"

TEST(CustomTypes, VectorArithmetic) {
    ncarray::Float2 a { 1.0f, 2.0f };
    ncarray::Float2 b { 3.0f, 4.0f };

    auto c = a + b;
    EXPECT_FLOAT_EQ(c.x, 4.0f);
    EXPECT_FLOAT_EQ(c.y, 6.0f);

    auto d = a * 2.0f;
    EXPECT_FLOAT_EQ(d.x, 2.0f);
    EXPECT_FLOAT_EQ(d.y, 4.0f);

    auto e = 3.0f * b;
    EXPECT_FLOAT_EQ(e.x, 9.0f);
    EXPECT_FLOAT_EQ(e.y, 12.0f);

    ncarray::Double2 f { 1.0, 2.0 };
    ncarray::Double2 g { 3.0, 4.0 };

    auto h = a + b;
    EXPECT_FLOAT_EQ(h.x, 4.0);
    EXPECT_FLOAT_EQ(h.y, 6.0);

    auto i = a * 2.0;
    EXPECT_FLOAT_EQ(i.x, 2.0);
    EXPECT_FLOAT_EQ(i.y, 4.0);

    auto j = 3.0 * b;
    EXPECT_FLOAT_EQ(j.x, 9.0);
    EXPECT_FLOAT_EQ(j.y, 12.0);
}

TEST(CustomTypes, CastingAndBroadcasting) {
  ncarray::Float4 vec4(1.23f);
  EXPECT_FLOAT_EQ(vec4.x, 1.23f);
  EXPECT_FLOAT_EQ(vec4.w, 1.23f);

  // Cast smaller to larger (Zero-fill check)
  ncarray::Float2 vec2 { 10.0f, 20.0f };
  ncarray::Float4 casted = static_cast<ncarray::Float4>(vec2);
  EXPECT_FLOAT_EQ(casted.x, 10.0f);
  EXPECT_FLOAT_EQ(casted.z, 0.0f); // Should be zero-filled

  ncarray::Float2 inf_vec { 1.0f, std::numeric_limits<float>::infinity() };
  EXPECT_FALSE(ncarray::op_traits<ncarray::Float2>::isfinite(inf_vec));

  ncarray::Double4 vec4_d(1.23f);
  EXPECT_FLOAT_EQ(vec4_d.x, 1.23);
  EXPECT_FLOAT_EQ(vec4_d.w, 1.23);

  ncarray::Double2 vec2_d { 10.0, 20.0 };
  ncarray::Double4 casted_d = static_cast<ncarray::Double4>(vec2_d);
  EXPECT_FLOAT_EQ(casted_d.x, 10.0);
  EXPECT_FLOAT_EQ(casted_d.z, 0.0);

  ncarray::Double2 inf_vec_d { 1.0, std::numeric_limits<double>::infinity() };
  EXPECT_FALSE(ncarray::op_traits<ncarray::Double2>::isfinite(inf_vec_d));
}
