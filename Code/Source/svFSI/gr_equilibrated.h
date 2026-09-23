/**
 * Copyright (c) Stanford University, The Regents of the University of
 * California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "stdafx.h"
#include <iomanip>
#include <iostream>
#include <limits>

#define _USE_MATH_DEFINES // to introduce pi constant (1/2)
#include <cassert>
#include <cmath>
#include <mat3d.h>
#include <math.h> // to introduce pi constant (2/2)
#include <stdafx.h>
#include <tens3d.h>
#include <tens4d.h>
#include <vec2d.h>
#include <vec3d.h>

#include "Array.h"
#include "ComMod.h"
#include "Tensor4.h"

#ifndef GR_EQUILIBRATED
#define GR_EQUILIBRATED

namespace gr_equilibrated_ns {

// When true, stress_tangent_ returns zero stress/tangent for an inverted
// element (J < 0) instead of throwing — set while dumping a crash-state VTU.
extern bool tolerate_negative_jacobian;

void stress_tangent_(const grModelType &grM, const double Fe[3][3],
                     const double time, const Vector<double> &eVWP,
                     Vector<double> &grInt, double S_out[3][3],
                     double CC_out[3][3][3][3], const bool eval_s = true,
                     const bool eval_cc = true);

void stress_tangent_stvk(const grModelType &grM, const double Fe[3][3],
                         const double time, const Vector<double> &eVWP,
                         Vector<double> &grInt, double S_out[3][3],
                         double CC_out[3][3][3][3], const bool eval_s = true,
                         const bool eval_cc = true);

}; // namespace gr_equilibrated_ns

#endif