/**
 * On-the-fly, per-node time-domain reduction of a scalar wall-shear-stress
 * magnitude signal over the trailing cycle_steps timesteps of a solver
 * invocation, driven by a runtime-configurable exprtk expression pair
 * (see ComMod.h's WssReductionType for the XML-sourced config). Replaces
 * writing one VTU per timestep purely to let a downstream Python driver
 * compute a time-domain WSS reduction (TAWSS, peak-to-peak amplitude, ...)
 * after the fact.
 */

#ifndef WSS_REDUCTION_H
#define WSS_REDUCTION_H

// Array.h/Vector.h use memcpy/memset without including <cstring>
// themselves; in the full solver build some earlier translation unit
// always supplies it first, but that's not guaranteed for every
// consumer of this header, so include it explicitly here.
#include <cstring>

#include "Array.h"

#include <string>

// Forward declared, not included: only wss_reduction.cpp's Simulation-
// dependent methods (added in a later phase) need the full type. Keeping
// it out of this header avoids pulling Simulation.h's dependency graph
// into every translation unit that includes wss_reduction.h.
class Simulation;

namespace wss_reduction {

class Accumulator
{
  public:
    /// @brief Per-node running-statistic rows in state_.
    enum StateRow { SUM = 0, MINV = 1, MAXV = 2, N = 3, NUM_STATE_ROWS = 4 };

    Accumulator();
    ~Accumulator();

    // Not copyable (owns a heap-allocated exprtk pImpl); not needed.
    Accumulator(const Accumulator&) = delete;
    Accumulator& operator=(const Accumulator&) = delete;

    /// @brief Dependency-free setup: parses update_expr/finalize_expr
    /// exactly once (parsing is the expensive part; per-node/per-timestep
    /// evaluation is then cheap) and allocates per-node state for nNodes
    /// nodes. Safe to call standalone, with no Simulation/mesh/MPI context
    /// -- this is what a unit test exercises directly.
    /// @throws std::runtime_error if either expression fails to parse.
    void init_core(int nNodes, const std::string& update_expr, const std::string& finalize_expr);

    /// @brief Per-node update: copies this node's running state into the
    /// expression's bound scratch variables, evaluates update_expr (whose
    /// ":=" assignments mutate those scratch variables in place), then
    /// copies the mutated values back into state_. O(1), no re-parsing.
    void update(int local_node_idx, double wss_mag);

    /// @brief Evaluates finalize_expr once per node, populating output().
    void finalize();

    /// @brief Zeros all per-node state (and output). Used at the start of
    /// a fresh accumulation window.
    void reset();

    int n_nodes() const { return nNodes_; }
    double output(int local_node_idx) const { return output_(0, local_node_idx); }
    double state(StateRow row, int local_node_idx) const { return state_(row, local_node_idx); }

    /// @brief Simulation-dependent setup for this invocation's accumulation
    /// window: resolves Wall_reduction_face_name to a mesh (via
    /// all_fun::find_face), throws if it doesn't resolve to face index 0
    /// (post::bpost's own hardcoded assumption), validates the other
    /// Wall_reduction_* fields (conditionally required only when enabled),
    /// computes this invocation's target stopTS/window-start independently
    /// of main.cpp's own locals (see main.cpp:100-150 vs. the unrelated
    /// STOP_SIM-file local shadowing "stopTS" at main.cpp:559), and calls
    /// init_core() (which starts state_/output_ zeroed). Then, only when
    /// continuing a previous simulation (com_mod.stFileFlag) and this
    /// invocation's windowStartCTS matches a valid <stFileName>_wss_
    /// reduction.bin sidecar's own stored windowStartCTS (i.e. we are
    /// resuming a crash *inside* the same still-open window, not starting
    /// a new one), loads that sidecar's saved per-node state_ over the
    /// freshly-zeroed one -- see load_restart_sidecar(). No-op (state stays
    /// zeroed) for every other case: fresh start, no sidecar found, or a
    /// sidecar whose window doesn't match (the normal case: a new FSG
    /// coupling invocation opening a new window).
    /// No-op entirely if com_mod.wssRed.enabled is false.
    void init(Simulation* simulation);

    /// @brief Restart-checkpoint hook, meant to be called alongside the
    /// existing output::write_restart() (same l1||l2 save cadence, main.cpp
    /// 594-596). Writes state_ to <stFileName>_wss_reduction.bin, one
    /// fixed-length record per rank at a rank-specific offset within the
    /// same shared file -- mirrors output::write_restart()/init_from_bin()'s
    /// own per-rank-record convention (output.cpp:283-285,
    /// initialize.cpp:118-121) exactly, so it needs no MPI gather: every
    /// rank already owns a stable partition, so it can write/read its own
    /// slice directly. No-op if disabled.
    void write_restart_sidecar(Simulation* simulation);

    /// @brief Per-timestep hook, meant to be called once per converged
    /// timestep (after the Newton loop, alongside the existing per-step
    /// outputs). Once cTS has entered the trailing accumulation window,
    /// computes this node's WSS magnitude via the existing post::bpost
    /// (already computed every VTU-saved timestep today for the WSS output
    /// group, so this adds no new Gauss-point/MPI cost) and folds it into
    /// the running per-node state. No-op if disabled or not yet in-window.
    void update_from_solution(Simulation* simulation);

    /// @brief End-of-invocation hook: finalizes per-node output, gathers it
    /// (and node coordinates) to rank 0 via the existing all_fun::global
    /// helper, and writes a single .vtu point cloud (0-based
    /// "GlobalNodeID" + "WSS_reduction" point-data arrays; no connectivity,
    /// since downstream consumption only needs the keyed scalar values --
    /// .vtu, not .vtp, since VtkVtpData doesn't implement Array<double>
    /// point data).
    /// No-op if disabled.
    void finalize_and_write(Simulation* simulation);

  private:
    // Attempts to load state_ from <stFileName>_wss_reduction.bin, per the
    // reset-vs-continue rule documented on init(). Returns true if this
    // rank's record was found, validated (magic/version/nNodes_ match, and
    // the sidecar's own stored windowStartCTS equals this invocation's
    // freshly-computed windowStartCTS_), and loaded; false otherwise (in
    // which case the caller leaves state_ at its freshly-zeroed value).
    bool load_restart_sidecar(Simulation* simulation);

    int nNodes_ = 0;
    Array<double> state_;   // (NUM_STATE_ROWS, nNodes_)
    Array<double> output_;  // (1, nNodes_)

    // exprtk plumbing is pImpl'd so exprtk.hpp (a ~47k-line template
    // header) is only ever included by wss_reduction.cpp, not by every
    // file that includes this header.
    struct ExprtkState;
    ExprtkState* expr_ = nullptr;

    bool enabled_ = false;
    int iM_ = -1;
    int stopTS_ = 0;
    int windowStartCTS_ = 0;
};

} // namespace wss_reduction

#endif
