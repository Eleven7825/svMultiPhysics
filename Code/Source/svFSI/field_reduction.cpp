// Dependency-free core of field_reduction::Accumulator: per-node,
// per-channel exprtk state management only (init_core/update/finalize/
// reset). No Simulation/mesh/MPI/VTK dependency, by design -- this is what
// test_field_reduction_standalone.cpp links against directly, without
// pulling in the rest of the solver. The Simulation-dependent methods
// (init/update_from_solution/finalize_and_write/write_restart_sidecar,
// plus the *_all() free functions) are defined in field_reduction_sim.cpp.

#include "field_reduction.h"

#include "exprtk.hpp"

#include <cmath>
#include <stdexcept>

namespace field_reduction {

struct Accumulator::ExprtkState
{
    // Persistent scratch variables. The symbol table binds expression
    // variable names to the *addresses* of these doubles once, at parse
    // time; update()/finalize() just copy values in/out and call
    // .value(), never re-parsing.
    double val = 0.0;
    double val_sum = 0.0;
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
  for (auto* e : channelExpr_) {
    delete e;
  }
}

void Accumulator::init_core(int nNodes, int nChannels, const std::string& update_expr_str,
    const std::string& finalize_expr_str, int bufferSteps, int vecDim)
{
  nNodes_ = nNodes;
  nChannels_ = nChannels;
  bufferSteps_ = bufferSteps;

  for (auto* e : channelExpr_) {
    delete e;
  }
  channelExpr_.assign(nChannels_, nullptr);
  channelState_.assign(nChannels_, Array<double>());

  output_.resize(nChannels_, nNodes_);
  output_ = 0.0;

  wssBuffer_.clear();
  if (bufferSteps_ > 0) {
    wssBuffer_.assign(bufferSteps_, Array<double>());
    for (auto& step : wssBuffer_) {
      step.resize(vecDim, nNodes_);
      step = 0.0;
    }
  }

  exprtk::parser<double> parser;

  for (int c = 0; c < nChannels_; c++) {
    channelState_[c].resize(NUM_STATE_ROWS, nNodes_);
    channelState_[c] = 0.0;

    auto* expr = new ExprtkState();
    channelExpr_[c] = expr;

    expr->symbol_table.add_variable("val", expr->val);
    // "sum" is exprtk's built-in vararg sum(...) function; a bound variable
    // of that name is silently shadowed/misparsed, so the exposed name here
    // is "val_sum" instead. Likewise "n" is exposed as-is (no known exprtk
    // builtin collides with it), but keep this comment as a warning for
    // anyone adding more bound variables: check against exprtk's reserved
    // function/keyword list first.
    expr->symbol_table.add_variable("val_sum", expr->val_sum);
    expr->symbol_table.add_variable("minv", expr->minv);
    expr->symbol_table.add_variable("maxv", expr->maxv);
    expr->symbol_table.add_variable("n", expr->cnt);
    expr->symbol_table.add_constants();

    expr->update_expr.register_symbol_table(expr->symbol_table);
    expr->finalize_expr.register_symbol_table(expr->symbol_table);

    // Same expression *text* compiled independently per channel -- each
    // channel needs its own persistent scratch variables/symbol table, but
    // parsing the same simple formula a handful of times (nChannels <= 3)
    // is still the cheap, one-time part of the design.
    if (!parser.compile(update_expr_str, expr->update_expr)) {
      throw std::runtime_error("field_reduction::Accumulator: failed to parse Update_expr '"
          + update_expr_str + "': " + parser.error());
    }
    if (!parser.compile(finalize_expr_str, expr->finalize_expr)) {
      throw std::runtime_error("field_reduction::Accumulator: failed to parse Finalize_expr '"
          + finalize_expr_str + "': " + parser.error());
    }
  }
}

void Accumulator::update(int a, int channel, double value)
{
  auto* expr = channelExpr_[channel];
  auto& state = channelState_[channel];

  expr->val_sum = state(SUM, a);
  expr->minv = state(MINV, a);
  expr->maxv = state(MAXV, a);
  expr->cnt = state(N, a);
  expr->val = value;

  expr->update_expr.value();

  state(SUM, a) = expr->val_sum;
  state(MINV, a) = expr->minv;
  state(MAXV, a) = expr->maxv;
  state(N, a) = expr->cnt;
}

void Accumulator::finalize()
{
  for (int c = 0; c < nChannels_; c++) {
    auto* expr = channelExpr_[c];
    auto& state = channelState_[c];

    for (int a = 0; a < nNodes_; a++) {
      expr->val_sum = state(SUM, a);
      expr->minv = state(MINV, a);
      expr->maxv = state(MAXV, a);
      expr->cnt = state(N, a);
      output_(c, a) = expr->finalize_expr.value();
    }
  }
}

void Accumulator::reset()
{
  for (auto& state : channelState_) {
    state = 0.0;
  }
  output_ = 0.0;
  combinedOutput_ = 0.0;
  for (auto& step : wssBuffer_) {
    step = 0.0;
  }
}

void Accumulator::record_raw(int step_idx, int component, int local_node_idx, double value)
{
  if (step_idx < 0 || step_idx >= bufferSteps_) {
    throw std::runtime_error("field_reduction::Accumulator::record_raw(): step_idx "
        + std::to_string(step_idx) + " out of range [0, " + std::to_string(bufferSteps_) + ").");
  }
  wssBuffer_[step_idx](component, local_node_idx) = value;
}

void Accumulator::combine_osi()
{
  if (nChannels_ != 4) {
    throw std::runtime_error("field_reduction::Accumulator::combine_osi(): requires exactly 4 "
        "internal channels (3 componentwise WSS + 1 magnitude), got "
        + std::to_string(nChannels_) + ".");
  }

  combinedOutput_.resize(1, nNodes_);

  for (int a = 0; a < nNodes_; a++) {
    double mx = output_(0, a);
    double my = output_(1, a);
    double mz = output_(2, a);
    double meanMag = output_(3, a);
    double vecMag = std::sqrt(mx*mx + my*my + mz*mz);
    // A (near-)zero mean magnitude means no measurable shear was ever seen
    // at this node -- OSI is undefined there, reported as 0 rather than
    // dividing by ~0.
    combinedOutput_(0, a) = (meanMag > 1e-12) ? 0.5 * (1.0 - vecMag / meanMag) : 0.0;
  }
}

void Accumulator::combine_transwss(const Array<double>& faceNormal)
{
  if (nChannels_ != 3) {
    throw std::runtime_error("field_reduction::Accumulator::combine_transwss(): requires exactly "
        "3 internal channels (componentwise WSS mean), got " + std::to_string(nChannels_) + ".");
  }
  if (bufferSteps_ <= 0 || wssBuffer_.empty()) {
    throw std::runtime_error("field_reduction::Accumulator::combine_transwss(): no raw WSS "
        "buffer was allocated (init_core() must be called with bufferSteps > 0).");
  }
  if (faceNormal.nrows() != 3 || faceNormal.ncols() != nNodes_) {
    throw std::runtime_error("field_reduction::Accumulator::combine_transwss(): faceNormal must "
        "be (3, n_nodes()), got (" + std::to_string(faceNormal.nrows()) + ", "
        + std::to_string(faceNormal.ncols()) + ").");
  }

  combinedOutput_.resize(1, nNodes_);

  for (int a = 0; a < nNodes_; a++) {
    double mx = output_(0, a);
    double my = output_(1, a);
    double mz = output_(2, a);

    double nx = faceNormal(0, a);
    double ny = faceNormal(1, a);
    double nz = faceNormal(2, a);

    // e = normalize(faceNormal x mean_vec) -- the transverse in-plane
    // direction perpendicular to the time-averaged WSS.
    double ex = ny*mz - nz*my;
    double ey = nz*mx - nx*mz;
    double ez = nx*my - ny*mx;
    double eNorm = std::sqrt(ex*ex + ey*ey + ez*ez);

    if (eNorm < 1e-12) {
      // faceNormal and mean_vec are (near-)parallel (or mean_vec is
      // ~zero) -- no well-defined transverse direction at this node.
      combinedOutput_(0, a) = 0.0;
      continue;
    }
    ex /= eNorm;
    ey /= eNorm;
    ez /= eNorm;

    double sum = 0.0;
    for (int t = 0; t < bufferSteps_; t++) {
      const auto& step = wssBuffer_[t];
      double dot = step(0, a)*ex + step(1, a)*ey + step(2, a)*ez;
      sum += std::fabs(dot);
    }
    combinedOutput_(0, a) = sum / bufferSteps_;
  }
}

} // namespace field_reduction
