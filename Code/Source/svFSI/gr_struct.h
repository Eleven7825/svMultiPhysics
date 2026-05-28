/* Copyright (c) Stanford University, The Regents of the University of
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

#ifndef GR_H
#define GR_H

#include <set>

#include "ComMod.h"
#include "Simulation.h"
#include "gr_equilibrated.h"
#include "mat_fun.h"
// #include "mat_models_carray.h"

namespace gr {

void construct_gr(ComMod &com_mod, const mshType &lM, const Array<double> &Dg,
                  bool eval_fd = true);

void eval_gr(const int &e, ComMod &com_mod, const mshType &lM,
             const Array<double> &Dg, Vector<int> &ptr, Array<double> &lR,
             Array3<double> &lK, const bool eval_s = true,
             const bool eval_cc = true, const double *F_bar_e = nullptr);

void struct_3d_carray(ComMod &com_mod, const int eNoN, const double w,
                      const Vector<double> &N, const Array<double> &Nx,
                      const Array<double> &dl, Vector<double> &gr_int_g,
                      Array<double> &gr_props_l, Array<double> &lR,
                      Array3<double> &lK, const bool eval_s = true,
                      const bool eval_cc = true, const double *F_bar_e = nullptr);

template <size_t N>
void cc_to_voigt_carray(const double CC[N][N][N][N], double Dm[2 * N][2 * N]) {
  if (N == 3) {
    Dm[0][0] = CC[0][0][0][0];
    Dm[0][1] = CC[0][0][1][1];
    Dm[0][2] = CC[0][0][2][2];
    Dm[0][3] = CC[0][0][0][1];
    Dm[0][4] = CC[0][0][1][2];
    Dm[0][5] = CC[0][0][2][0];

    Dm[1][1] = CC[1][1][1][1];
    Dm[1][2] = CC[1][1][2][2];
    Dm[1][3] = CC[1][1][0][1];
    Dm[1][4] = CC[1][1][1][2];
    Dm[1][5] = CC[1][1][2][0];

    Dm[2][2] = CC[2][2][2][2];
    Dm[2][3] = CC[2][2][0][1];
    Dm[2][4] = CC[2][2][1][2];
    Dm[2][5] = CC[2][2][2][0];

    Dm[3][3] = CC[0][1][0][1];
    Dm[3][4] = CC[0][1][1][2];
    Dm[3][5] = CC[0][1][2][0];

    Dm[4][4] = CC[1][2][1][2];
    Dm[4][5] = CC[1][2][2][0];

    Dm[5][5] = CC[2][0][2][0];

    // Upper triangle
    Dm[1][0] = CC[1][1][0][0];

    Dm[2][0] = CC[2][2][0][0];
    Dm[2][1] = CC[2][2][1][1];

    Dm[3][0] = CC[0][1][0][0];
    Dm[3][1] = CC[0][1][1][1];
    Dm[3][2] = CC[0][1][2][2];

    Dm[4][0] = CC[1][2][0][0];
    Dm[4][1] = CC[1][2][1][1];
    Dm[4][2] = CC[1][2][2][2];
    Dm[4][3] = CC[1][2][0][1];

    Dm[5][0] = CC[2][0][0][0];
    Dm[5][1] = CC[2][0][1][1];
    Dm[5][2] = CC[2][0][2][2];
    Dm[5][3] = CC[2][0][0][1];
    Dm[5][4] = CC[2][0][1][2];
  }
}

template <size_t N>
void get_pk2cc(const ComMod &com_mod, const dmnType &lDmn, const double F[N][N],
               Vector<double> &gr_int, const Vector<double> &gr_props,
               double S[N][N], double Dm[2 * N][2 * N],
               const bool eval_s = true, const bool eval_cc = true) {
  using namespace consts;
  using namespace mat_fun;
  using namespace utils;

  const auto &stM = lDmn.stM;
  const auto &grM = lDmn.grM;
  const int nsd = com_mod.nsd;

  double CC[N][N][N][N];

  switch (stM.isoType) {
  case ConstitutiveModelType::GR_equi: {
    gr_equilibrated_ns::stress_tangent_(grM, F, com_mod.time, gr_props, gr_int,
                                        S, CC, eval_s, eval_cc);
  } break;

  default:
    throw std::runtime_error("Undefined material constitutive model.");
  }

  // Convert to Voigt Notation
  cc_to_voigt_carray<N>(CC, Dm);
}

}; // namespace gr

#endif
