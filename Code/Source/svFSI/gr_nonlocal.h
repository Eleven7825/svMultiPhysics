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
 * THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef GR_NONLOCAL_H
#define GR_NONLOCAL_H

#include <array>
#include <vector>

#include "Array.h"
#include "ComMod.h"

namespace gr_nonlocal_ns {

// Compute element centroid (theta, z, ro) in the circumferential-axial plane.
// theta is in [-pi, pi] using atan2 to handle the full cylinder.
// gr_props indices 3-5 hold the reference coordinates X.
void compute_elem_coords(const mshType& lM, const grModelType& grM, int e,
                         double& theta_e, double& z_e, double& ro_e);

// Compute the volume-weighted average deformation gradient for element e
// over all its Gauss points. F_elem_out is a flat row-major 3x3 array (9 values).
void compute_elem_F(const mshType& lM, const ComMod& com_mod,
                    const Array<double>& Dg, int e, double F_elem_out[9]);

// Boundary correction factor gamma(x) from Eq. C.5.
// d_x: distance to nearest axial end boundary
// t, beta, l: filter parameters
double compute_gamma(double d_x, double t, double beta, double l);

// For every element, compute the spatially averaged F_bar using the
// boundary-corrected Gaussian kernel (Eq. C.3, C.5) restricted to the
// circumferential-axial plane. Each element in F_bar_elem receives
// a flat row-major 3x3 averaged deformation gradient.
void compute_nonlocal_F(const mshType& lM, const grModelType& grM,
                        const std::vector<std::array<double,9>>& F_elem,
                        const std::vector<double>& theta_elem,
                        const std::vector<double>& z_elem,
                        const std::vector<double>& ro_elem,
                        std::vector<std::array<double,9>>& F_bar_elem);

} // namespace gr_nonlocal_ns

#endif
