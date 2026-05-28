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

// Non-local spatial filter for the G&R deformation gradient.
// Implements the boundary-corrected Gaussian kernel from Appendix C
// (Eq. C.3, C.5) applied in the circumferential-axial (theta, z) plane.
// Each mesh layer (radial band) is filtered independently.

#include "gr_nonlocal.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "Array.h"
#include "ComMod.h"
#include "nn.h"
#include "mat_fun_carray.h"

namespace gr_nonlocal_ns {

// ---------------------------------------------------------------------------
// compute_elem_coords
// ---------------------------------------------------------------------------
// Computes the element centroid's (theta, z, ro) coordinates in the
// circumferential-axial parameterisation used throughout gr_equilibrated.cpp.
//
// Reference coordinates are stored in lM.gr_props at indices 3, 4, 5.
// The centerline follows Xcl = (curve/2*(1-cos(2pi*z/lo)), 0, z).
// theta = atan2(NX.x, -NX.y) gives the full [-pi, pi] azimuthal range.
void compute_elem_coords(const mshType& lM, const grModelType& grM, int e,
                         double& theta_e, double& z_e, double& ro_e)
{
  const int eNoN = lM.eNoN;
  const double lo = grM.lo * grM.mult;

  double X[3] = {0.0, 0.0, 0.0};
  for (int a = 0; a < eNoN; a++) {
    const int Ac = lM.IEN(a, e);
    X[0] += lM.gr_props(3, Ac);
    X[1] += lM.gr_props(4, Ac);
    X[2] += lM.gr_props(5, Ac);
  }
  X[0] /= eNoN;
  X[1] /= eNoN;
  X[2] /= eNoN;

  // Centerline position (matches gr_equilibrated.cpp line 116)
  const double Xcl_x = grM.curve / 2.0 * (1.0 - std::cos(2.0 * M_PI * X[2] / lo));
  const double Xcl_y = 0.0;
  const double Xcl_z = X[2];

  double NX[3] = {X[0] - Xcl_x, X[1] - Xcl_y, X[2] - Xcl_z};
  ro_e = std::sqrt(NX[0]*NX[0] + NX[1]*NX[1] + NX[2]*NX[2]);

  if (ro_e < 1.0e-14)
    throw std::runtime_error("[gr_nonlocal] Element centroid on centerline; cannot compute theta");

  NX[0] /= ro_e;
  NX[1] /= ro_e;
  NX[2] /= ro_e;

  // Full azimuthal angle in [-pi, pi] (atan2 handles the full cylinder)
  theta_e = std::atan2(NX[0], -NX[1]);
  z_e     = X[2];
}

// ---------------------------------------------------------------------------
// compute_elem_F
// ---------------------------------------------------------------------------
// Computes the volume-weighted average deformation gradient for element e.
// F_avg = sum_g(w_g * Jac_g * F_g) / sum_g(w_g * Jac_g)
// F_elem_out is written as a flat row-major [i*3+j] array.
void compute_elem_F(const mshType& lM, const ComMod& com_mod,
                    const Array<double>& Dg, int e, double F_elem_out[9])
{
  const int nsd  = com_mod.nsd;   // == 3
  const int tDof = com_mod.tDof;
  const int eNoN = lM.eNoN;

  // Displacement DOF start index
  const int cEq = com_mod.cEq;
  const int s   = com_mod.eq[cEq].s;
  const int idx[3] = {s, s+1, s+2};

  // Gather element node coordinates and displacements
  Array<double> xl(nsd, eNoN);
  Array<double> dl(tDof, eNoN);
  for (int a = 0; a < eNoN; a++) {
    const int Ac = lM.IEN(a, e);
    for (int i = 0; i < nsd; i++)
      xl(i, a) = com_mod.x(i, Ac);
    for (int i = 0; i < tDof; i++)
      dl(i, a) = Dg(i, Ac);
  }

  double F_avg[3][3] = {};
  double w_total = 0.0;

  Array<double> Nx(nsd, eNoN);
  Array<double> ksix(nsd, nsd);
  double Jac = 0.0;

  for (int g = 0; g < lM.nG; g++) {
    // Shape function gradients in reference config
    if (g == 0 || !lM.lShpF) {
      auto Nx_g = lM.Nx.slice(g);
      nn::gnn(eNoN, nsd, nsd, Nx_g, xl, Nx, Jac, ksix);
    }
    const double w = lM.w(g) * Jac;

    // Build deformation gradient F = I + Grad(u)
    double F[3][3] = {{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
    for (int a = 0; a < eNoN; a++)
      for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
          F[row][col] += Nx(col, a) * dl(idx[row], a);

    for (int row = 0; row < 3; row++)
      for (int col = 0; col < 3; col++)
        F_avg[row][col] += w * F[row][col];
    w_total += w;
  }

  if (w_total < 1.0e-30)
    throw std::runtime_error("[gr_nonlocal] Zero volume for element in compute_elem_F");

  for (int row = 0; row < 3; row++)
    for (int col = 0; col < 3; col++)
      F_elem_out[row*3 + col] = F_avg[row][col] / w_total;
}

// ---------------------------------------------------------------------------
// compute_gamma
// ---------------------------------------------------------------------------
// Boundary correction factor from Eq. C.5:
//   gamma(x) = 1                           if d(x) >= t*l
//   gamma(x) = (1-beta)/(t*l) * d(x) + beta   if d(x) <  t*l
double compute_gamma(double d_x, double t, double beta, double l)
{
  const double tl = t * l;
  if (d_x >= tl)
    return 1.0;
  if (tl < 1.0e-30)
    return beta;
  return (1.0 - beta) / tl * d_x + beta;
}

// ---------------------------------------------------------------------------
// compute_nonlocal_F
// ---------------------------------------------------------------------------
// For each element e, computes the spatially averaged deformation gradient
// F_bar(x) = sum_{|x-xi|<=R} omega'(x,xi) * F(xi)
// using the boundary-corrected Gaussian kernel (Eq. C.3, C.5) restricted
// to the circumferential-axial plane. Elements in different radial layers
// (|ro_e - ro_k| > ro_tol) are excluded from each other's neighbourhood.
// The hard cutoff radius is R = 3 * gamma(x) * l.
void compute_nonlocal_F(const mshType& lM, const grModelType& grM,
                        const std::vector<std::array<double,9>>& F_elem,
                        const std::vector<double>& theta_elem,
                        const std::vector<double>& z_elem,
                        const std::vector<double>& ro_elem,
                        std::vector<std::array<double,9>>& F_bar_elem)
{
  const int    nEl   = lM.nEl;
  const double l     = grM.l_nl;
  const double beta  = grM.beta_nl;
  const double t     = grM.t_nl;
  const double ro_tol = grM.ro_tol;
  const double lo    = grM.lo * grM.mult;

  F_bar_elem.resize(nEl);

  for (int e = 0; e < nEl; e++) {
    const double z_e     = z_elem[e];
    const double theta_e = theta_elem[e];
    const double ro_e    = ro_elem[e];

    // Distance to nearest axial end boundary
    const double d_x    = std::min(z_e, lo - z_e);
    const double gamma_x = compute_gamma(d_x, t, beta, l);
    const double eff_l   = gamma_x * l;
    const double R_cut   = 3.0 * eff_l;

    double w_sum = 0.0;
    double F_bar[9] = {};

    for (int k = 0; k < nEl; k++) {
      // Radial layer check: skip elements in different layers
      if (std::abs(ro_elem[k] - ro_e) > ro_tol)
        continue;

      // Circumferential arc-length distance (handle [-pi,pi] wraparound)
      double dtheta = theta_elem[k] - theta_e;
      while (dtheta >  M_PI) dtheta -= 2.0 * M_PI;
      while (dtheta < -M_PI) dtheta += 2.0 * M_PI;
      const double d_arc = ro_e * std::abs(dtheta);
      const double d_z   = std::abs(z_elem[k] - z_e);
      const double dist  = std::sqrt(d_arc*d_arc + d_z*d_z);

      if (dist > R_cut)
        continue;

      const double w = std::exp(-0.5 * (dist / eff_l) * (dist / eff_l));
      w_sum += w;
      for (int c = 0; c < 9; c++)
        F_bar[c] += w * F_elem[k][c];
    }

    if (w_sum > 0.0) {
      for (int c = 0; c < 9; c++)
        F_bar_elem[e][c] = F_bar[c] / w_sum;
    } else {
      // Fallback: no neighbours found, use element's own F
      F_bar_elem[e] = F_elem[e];
    }
  }
}

} // namespace gr_nonlocal_ns
