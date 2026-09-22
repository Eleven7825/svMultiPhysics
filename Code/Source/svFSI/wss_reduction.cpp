#include "wss_reduction.h"

#include "exprtk.hpp"

#include "Simulation.h"
#include "all_fun.h"
#include "post.h"
#include "VtkData.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace wss_reduction {

struct Accumulator::ExprtkState
{
    // Persistent scratch variables. The symbol table binds expression
    // variable names to the *addresses* of these doubles once, at parse
    // time; update()/finalize() just copy values in/out and call
    // .value(), never re-parsing.
    double wss_mag = 0.0;
    double wss_sum = 0.0;
    double minv = 0.0;
    double maxv = 0.0;
    double cnt = 0.0;

    exprtk::symbol_table<double> symbol_table;
    exprtk::expression<double> update_expr;
    exprtk::expression<double> finalize_expr;
};

Accumulator::Accumulator() {}

Accumulator::~Accumulator()
{
  delete expr_;
}

void Accumulator::init_core(int nNodes, const std::string& update_expr_str, const std::string& finalize_expr_str)
{
  nNodes_ = nNodes;
  state_.resize(NUM_STATE_ROWS, nNodes_);
  output_.resize(1, nNodes_);
  state_ = 0.0;
  output_ = 0.0;

  delete expr_;
  expr_ = new ExprtkState();

  expr_->symbol_table.add_variable("wss_mag", expr_->wss_mag);
  // "sum" is exprtk's built-in vararg sum(...) function; a bound variable
  // of that name is silently shadowed/misparsed, so the exposed name here
  // is "wss_sum" instead. Likewise "n" is exposed as-is (no known exprtk
  // builtin collides with it), but keep this comment as a warning for
  // anyone adding more bound variables: check against exprtk's reserved
  // function/keyword list first.
  expr_->symbol_table.add_variable("wss_sum", expr_->wss_sum);
  expr_->symbol_table.add_variable("minv", expr_->minv);
  expr_->symbol_table.add_variable("maxv", expr_->maxv);
  expr_->symbol_table.add_variable("n", expr_->cnt);
  expr_->symbol_table.add_constants();

  expr_->update_expr.register_symbol_table(expr_->symbol_table);
  expr_->finalize_expr.register_symbol_table(expr_->symbol_table);

  exprtk::parser<double> parser;

  if (!parser.compile(update_expr_str, expr_->update_expr)) {
    throw std::runtime_error("wss_reduction::Accumulator: failed to parse Wall_reduction_update_expr '"
        + update_expr_str + "': " + parser.error());
  }
  if (!parser.compile(finalize_expr_str, expr_->finalize_expr)) {
    throw std::runtime_error("wss_reduction::Accumulator: failed to parse Wall_reduction_finalize_expr '"
        + finalize_expr_str + "': " + parser.error());
  }
}

void Accumulator::update(int a, double wss_mag)
{
  expr_->wss_sum = state_(SUM, a);
  expr_->minv = state_(MINV, a);
  expr_->maxv = state_(MAXV, a);
  expr_->cnt = state_(N, a);
  expr_->wss_mag = wss_mag;

  expr_->update_expr.value();

  state_(SUM, a) = expr_->wss_sum;
  state_(MINV, a) = expr_->minv;
  state_(MAXV, a) = expr_->maxv;
  state_(N, a) = expr_->cnt;
}

void Accumulator::finalize()
{
  for (int a = 0; a < nNodes_; a++) {
    expr_->wss_sum = state_(SUM, a);
    expr_->minv = state_(MINV, a);
    expr_->maxv = state_(MAXV, a);
    expr_->cnt = state_(N, a);
    output_(0, a) = expr_->finalize_expr.value();
  }
}

void Accumulator::reset()
{
  state_ = 0.0;
  output_ = 0.0;
}

void Accumulator::init(Simulation* simulation)
{
  auto& com_mod = simulation->com_mod;
  auto& wssRed = com_mod.wssRed;

  enabled_ = wssRed.enabled;
  if (!enabled_) {
    return;
  }

  if (wssRed.faceName.empty()) {
    throw std::runtime_error("wss_reduction::Accumulator: Wall_reduction_enabled is true but "
        "Wall_reduction_face_name is empty.");
  }
  if (wssRed.cycleSteps <= 0) {
    throw std::runtime_error("wss_reduction::Accumulator: Wall_reduction_enabled is true but "
        "Wall_reduction_cycle_steps must be > 0 (got " + std::to_string(wssRed.cycleSteps) + ").");
  }
  if (wssRed.updateExpr.empty() || wssRed.finalizeExpr.empty()) {
    throw std::runtime_error("wss_reduction::Accumulator: Wall_reduction_enabled is true but "
        "Wall_reduction_update_expr/Wall_reduction_finalize_expr must both be non-empty.");
  }

  int iFa = -1;
  all_fun::find_face(com_mod.msh, wssRed.faceName, iM_, iFa);
  if (iFa != 0) {
    throw std::runtime_error("wss_reduction::Accumulator: Wall_reduction_face_name '" + wssRed.faceName
        + "' resolves to face index " + std::to_string(iFa) + " of its mesh, but post::bpost only "
        "computes WSS on face index 0.");
  }

  const auto& msh = com_mod.msh[iM_];
  init_core(msh.nNo, wssRed.updateExpr, wssRed.finalizeExpr);

  // This invocation's target final cTS, computed independently of
  // main.cpp's own local of the same name -- see main.cpp:100,148-150 (the
  // real one, in scope for the whole function) vs. the unrelated
  // STOP_SIM-file local that shadows it starting at main.cpp:559. Mirrors
  // main.cpp's own "int minTS = cTS + newTS; nTS = std::max(nTS, minTS);"
  // computed once, before the outer loop begins.
  stopTS_ = std::max(com_mod.nTS, com_mod.cTS + com_mod.newTS);
  windowStartCTS_ = stopTS_ - wssRed.cycleSteps + 1;
}

void Accumulator::update_from_solution(Simulation* simulation)
{
  if (!enabled_) {
    return;
  }

  auto& com_mod = simulation->com_mod;

  if (com_mod.cTS < windowStartCTS_) {
    return;
  }

  auto& msh = com_mod.msh[iM_];
  Array<double> tmpV(consts::maxNSD, msh.nNo);
  post::bpost(simulation, msh, tmpV, com_mod.Yn, com_mod.Dn, consts::OutputType::outGrp_WSS);

  const int nsd = com_mod.nsd;

  for (int a = 0; a < msh.nNo; a++) {
    double mag2 = 0.0;
    for (int i = 0; i < nsd; i++) {
      mag2 += tmpV(i,a) * tmpV(i,a);
    }
    update(a, std::sqrt(mag2));
  }
}

void Accumulator::finalize_and_write(Simulation* simulation)
{
  if (!enabled_) {
    return;
  }

  finalize();

  auto& com_mod = simulation->com_mod;
  auto& cm_mod = simulation->cm_mod;
  auto& cm = com_mod.cm;
  auto& msh = com_mod.msh[iM_];

  // msh.x is not a reliable (nsd, msh.nNo) buffer to gather directly --
  // vtk_xml::write_vtu_debug's own precedent builds its local coordinate
  // array explicitly from com_mod.x (the authoritative, always-current
  // per-rank coordinate array) via msh.gN, so mirror that here.
  const int nsd = com_mod.nsd;
  Array<double> x(nsd, msh.nNo);
  for (int a = 0; a < msh.nNo; a++) {
    int Ac = msh.gN(a);
    for (int i = 0; i < nsd; i++) {
      x(i,a) = com_mod.x(i,Ac);
    }
  }

  // Gather local (per-rank), local-node-indexed arrays to rank 0, reindexed
  // to global node order -- the same helper used to gather point-data
  // arrays for the ordinary VTU writer (vtk_xml.cpp's "d.gx =
  // all_fun::global(...)"). Returns an empty array on non-master ranks.
  Array<double> gx = all_fun::global(com_mod, cm_mod, msh, x);
  Array<double> gOut = all_fun::global(com_mod, cm_mod, msh, output_);

  if (cm.slv(cm_mod)) {
    return;
  }

  const int gnNo = msh.gnNo;
  // VtkVtpData only implements set_point_data() for Vector<int> (used for
  // face id arrays like GlobalNodeID/GlobalElementID); its Array<double>
  // overload is an unimplemented stub that throws. VtkVtuData is the
  // reverse: Array<double>/Array<int> are implemented, Vector<int> is the
  // stub. Since WSS_reduction is a double field, the output must be a
  // .vtu (see Wall_reduction_output_file_path's default), and
  // GlobalNodeID must go through the Array<int> overload, not Vector<int>.
  Array<int> globalNodeId(1, gnNo);
  for (int a = 0; a < gnNo; a++) {
    globalNodeId(0,a) = a;
  }

  // One degenerate 1-node (VTK_VERTEX) cell per point. Not needed for
  // set_points()/set_point_data() themselves, but a VTU with zero cells
  // trips up common readers (confirmed: meshio's appended-data decoder
  // throws on an empty cell block) even though VTK's own writer accepts
  // it -- and Phase 5's downstream Python consumption needs this file to
  // be ordinarily readable, not just structurally legal.
  Array<int> vertexConn(1, gnNo);
  for (int a = 0; a < gnNo; a++) {
    vertexConn(0,a) = a;
  }

  auto vtk_writer = VtkData::create_writer(com_mod.wssRed.outputFilePath);
  vtk_writer->set_points(gx);
  vtk_writer->set_connectivity(nsd, vertexConn);
  vtk_writer->set_point_data("GlobalNodeID", globalNodeId);
  vtk_writer->set_point_data("WSS_reduction", gOut);
  vtk_writer->write();
  delete vtk_writer;
}

} // namespace wss_reduction
