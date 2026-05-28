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

// Functions for solving nonlinear structural mechanics
// problems (pure displacement-based formulation).
//
// Replicates the Fortran functions in 'STRUCT.f'.

#include "gr_struct.h"
#include "gr_nonlocal.h"
#include "sv_struct.h"

#include "all_fun.h"
#include "consts.h"
#include "gr_equilibrated.h"
#include "lhsa.h"
#include "mat_fun.h"
#include "mat_fun_carray.h"
#include "mat_models.h"
#include "nn.h"
#include "utils.h"
#include "vtk_xml.h"

#ifdef WITH_TRILINOS
#include "trilinos_linear_solver.h"
#endif

namespace gr {

/// @brief Loop solid elements and assemble into global matrices
void construct_gr(ComMod &com_mod, const mshType &lM, const Array<double> &Dg,
                  bool eval_fd) {
  using namespace consts;

  // Get dimensions
  const int eNoN = lM.eNoN;
  const int dof = com_mod.dof;

  // Initialize residual and tangent
  Vector<int> ptr(eNoN);
  Array<double> lR(dof, eNoN);
  Array3<double> lK(dof * dof, eNoN, eNoN);

  // Make editable copy
  Array<double> e_Dg(Dg);

  // Retrieve G&R model parameters for the current equation/domain
  const int cEq  = com_mod.cEq;
  const int cDmn = com_mod.cDmn;
  const auto &grM = com_mod.eq[cEq].dmn[cDmn].grM;

  // Non-local pre-pass: compute element-average F and spatially averaged F_bar
  std::vector<std::array<double,9>> F_elem, F_bar_elem;
  if (grM.use_nonlocal) {
    const int nEl = lM.nEl;
    F_elem.resize(nEl);
    F_bar_elem.resize(nEl);

    std::vector<double> theta_elem(nEl), z_elem(nEl), ro_elem(nEl);
    for (int e = 0; e < nEl; e++) {
      gr_nonlocal_ns::compute_elem_coords(lM, grM, e,
                                          theta_elem[e], z_elem[e], ro_elem[e]);
      gr_nonlocal_ns::compute_elem_F(lM, com_mod, Dg, e, F_elem[e].data());
    }
    gr_nonlocal_ns::compute_nonlocal_F(lM, grM, F_elem, theta_elem, z_elem,
                                       ro_elem, F_bar_elem);
  }

  // Loop over all elements of mesh
  for (int e = 0; e < lM.nEl; e++) {
    // Reset
    ptr = 0;
    lR = 0.0;
    lK = 0.0;

    const double *F_bar_e = grM.use_nonlocal ? F_bar_elem[e].data() : nullptr;

    // Update G&R internal variables
    eval_gr(e, com_mod, lM, Dg, ptr, lR, lK, false, false, F_bar_e);

    // Compute stress and tangent
    eval_gr(e, com_mod, lM, Dg, ptr, lR, lK, true, true, F_bar_e);

    // Assemble into global residual and tangent
    lhsa_ns::do_assem(com_mod, eNoN, ptr, lK, lR);
  }
}

/// @brief
void eval_gr(const int &e, ComMod &com_mod, const mshType &lM,
             const Array<double> &Dg, Vector<int> &ptr, Array<double> &lR,
             Array3<double> &lK, const bool eval_s, const bool eval_cc,
             const double *F_bar_e) {
  using namespace consts;

  const int nsd = com_mod.nsd;
  const int tDof = com_mod.tDof;
  int eNoN = lM.eNoN;
  assert(nsd == 3);

  // STRUCT: dof = nsd
  Vector<double> N(eNoN), gr_int_g(com_mod.nGrInt), gr_props_g(lM.n_gr_props);
  Array<double> xl(nsd, eNoN), dl(tDof, eNoN), Nx(nsd, eNoN),
      gr_props_l(lM.n_gr_props, eNoN);

  // Create local copies
  gr_int_g = 0.0;
  gr_props_l = 0.0;
  gr_props_g = 0.0;

  for (int a = 0; a < eNoN; a++) {
    int Ac = lM.IEN(a, e);
    ptr(a) = Ac;

    for (int i = 0; i < nsd; i++) {
      xl(i, a) = com_mod.x(i, Ac);
    }

    for (int i = 0; i < tDof; i++) {
      dl(i, a) = Dg(i, Ac);
    }

    if (lM.gr_props.size() != 0) {
      for (int igr = 0; igr < lM.n_gr_props; igr++) {
        gr_props_l(igr, a) = lM.gr_props(igr, Ac);
      }
    }
  }

  // Gauss integration
  double Jac{0.0};
  Array<double> ksix(nsd, nsd);

  for (int g = 0; g < lM.nG; g++) {
    if (g == 0 || !lM.lShpF) {
      auto Nx_g = lM.Nx.slice(g);
      nn::gnn(eNoN, nsd, nsd, Nx_g, xl, Nx, Jac, ksix);
      if (utils::is_zero(Jac)) {
        throw std::runtime_error("[construct_dsolid] Jacobian for element " +
                                 std::to_string(e) + " is < 0.");
      }
    }
    double w = lM.w(g) * Jac;
    N = lM.N.col(g);

    // Get internal growth and remodeling variables
    if (com_mod.grEq) {
      for (int i = 0; i < com_mod.nGrInt; i++) {
        gr_int_g(i) = com_mod.grInt(e, g, i);
      }
    }

    struct_3d_carray(com_mod, eNoN, w, N, Nx, dl, gr_int_g, gr_props_l, lR, lK,
                     eval_s, eval_cc, F_bar_e);

    // Set internal growth and remodeling variables
    if (com_mod.grEq) {
      // todo mrp089: add a function like rslice for vectors to Array3
      for (int i = 0; i < com_mod.nGrInt; i++) {
        com_mod.grInt(e, g, i) = gr_int_g(i);
      }
    }
  }
}

void struct_3d_carray(ComMod &com_mod, const int eNoN, const double w,
                      const Vector<double> &N, const Array<double> &Nx,
                      const Array<double> &dl, Vector<double> &gr_int_g,
                      Array<double> &gr_props_l, Array<double> &lR,
                      Array3<double> &lK, const bool eval_s,
                      const bool eval_cc, const double *F_bar_e) {
  using namespace consts;
  using namespace mat_fun;

  const int dof = com_mod.dof;
  int cEq = com_mod.cEq;
  auto &eq = com_mod.eq[cEq];
  int cDmn = com_mod.cDmn;
  auto &dmn = eq.dmn[cDmn];

  // Set parameters
  int i = eq.s;
  int j = i + 1;
  int k = j + 1;
  int indices[] = {i, j, k};

  // Inertia, body force and deformation tensor (F)
  double F[3][3] = {};
  Vector<double> gr_props_g(gr_props_l.nrows());

  F[0][0] = 1.0;
  F[1][1] = 1.0;
  F[2][2] = 1.0;

  for (int a = 0; a < eNoN; a++) {
    // Deformation gradient
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        F[row][col] += Nx(col, a) * dl(indices[row], a);
      }
    }

    // G&R material properties
    for (int igr = 0; igr < gr_props_l.nrows(); igr++) {
      gr_props_g(igr) += gr_props_l(igr, a) * N(a);
    }
  }

  double Jac = mat_fun_carray::mat_det<3>(F);
  double Fi[3][3];
  mat_fun_carray::mat_inv<3>(F, Fi);

  // Initialize tensor indexing.
  mat_fun_carray::ten_init(3);

  // 2nd Piola-Kirchhoff tensor (S) and material stiffness tensor in
  // Voigt notationa (Dm)
  double S[3][3];
  double Dm[6][6];

  // Use non-local averaged F_bar for material evaluation if provided.
  // Kinematics (P = F*S, Bm, NxSNx) continue to use the original F.
  double F_mat[3][3];
  if (F_bar_e != nullptr) {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        F_mat[i][j] = F_bar_e[i*3 + j];
  } else {
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        F_mat[i][j] = F[i][j];
  }
  get_pk2cc<3>(com_mod, dmn, F_mat, gr_int_g, gr_props_g, S, Dm, eval_s, eval_cc);

  if (!eval_s && !eval_cc) {
    return;
  }

  // 1st Piola-Kirchhoff tensor (P)
  double P[3][3];
  mat_fun_carray::mat_mul<3>(F, S, P);

  // Local residual
  for (int a = 0; a < eNoN; a++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        lR(i, a) += w * Nx(j, a) * P[i][j];
      }
    }
  }

  // Skip evaluation if tangent not required
  if (!eval_cc) {
    return;
  }

  // Auxilary quantities for computing stiffness tensor
  Array3<double> Bm(6, 3, eNoN);
  for (int a = 0; a < eNoN; a++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        Bm(i, j, a) = Nx(i, a) * F[j][i];
      }
    }

    Bm(3, 0, a) = (Nx(0, a) * F[0][1] + F[0][0] * Nx(1, a));
    Bm(3, 1, a) = (Nx(0, a) * F[1][1] + F[1][0] * Nx(1, a));
    Bm(3, 2, a) = (Nx(0, a) * F[2][1] + F[2][0] * Nx(1, a));

    Bm(4, 0, a) = (Nx(1, a) * F[0][2] + F[0][1] * Nx(2, a));
    Bm(4, 1, a) = (Nx(1, a) * F[1][2] + F[1][1] * Nx(2, a));
    Bm(4, 2, a) = (Nx(1, a) * F[2][2] + F[2][1] * Nx(2, a));

    Bm(5, 0, a) = (Nx(2, a) * F[0][0] + F[0][2] * Nx(0, a));
    Bm(5, 1, a) = (Nx(2, a) * F[1][0] + F[1][2] * Nx(0, a));
    Bm(5, 2, a) = (Nx(2, a) * F[2][0] + F[2][2] * Nx(0, a));
  }

  // Local stiffness tensor
  double NxSNx, BmDBm;
  Array<double> DBm(6, 3);

  for (int b = 0; b < eNoN; b++) {
    for (int a = 0; a < eNoN; a++) {
      // Geometric stiffness
      NxSNx = 0.0;
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          NxSNx += Nx(i, a) * S[i][j] * Nx(j, b);
        }
      }

      // Material Stiffness (Bt*D*B)
      mat_fun_carray::mat_mul6x3<3>(Dm, Bm.rslice(b), DBm);
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          BmDBm = 0.0;
          for (int k = 0; k < 6; k++) {
            BmDBm += Bm(k, i, a) * DBm(k, j);
          }
          int index = i * dof + j;
          lK(index, a, b) += w * BmDBm;
          if (index == 0 || index == 4 || index == 8) {
            lK(index, a, b) += w * NxSNx;
          }
        }
      }
    }
  }
}

}; // namespace gr