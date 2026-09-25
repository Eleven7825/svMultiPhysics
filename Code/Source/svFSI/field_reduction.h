/**
 * On-the-fly, per-node, multi-channel time-domain reduction of a scalar or
 * vector field over the trailing cycle_steps timesteps of a solver
 * invocation, driven by a runtime-configurable exprtk expression pair (see
 * ComMod.h's ReductionConfig for the XML-sourced config, populated from one
 * repeatable <Add_reduction> block). Replaces writing one VTU per timestep
 * purely to let a downstream Python driver compute a time-domain reduction
 * (TAWSS, peak-to-peak amplitude, time-average, ...) after the fact.
 *
 * One Accumulator instance corresponds to one <Add_reduction> block. A
 * block targets either:
 *   - Field=WSS, Scope=face: wall shear stress on one face, reduced either
 *     magnitude-first (1 channel) or componentwise (nsd channels), per a
 *     user-supplied Update_expr/Finalize_expr pair.
 *   - Field=Velocity|Pressure, Scope=volume: read directly from com_mod.Yn
 *     over the whole mesh, reduced either magnitude-first (1 channel) or
 *     componentwise (nsd channels for Velocity, 1 for Pressure), also via
 *     a user-supplied Update_expr/Finalize_expr pair.
 *   - Field=OSI, Scope=face: Oscillatory Shear Index, a *fixed* formula
 *     (no Update_expr/Finalize_expr) computed from 4 internal channels
 *     (3 componentwise WSS + 1 magnitude) combined at finalize() via
 *     combine_osi() -- see that method's doc comment for the formula.
 */

#ifndef FIELD_REDUCTION_H
#define FIELD_REDUCTION_H

// Array.h/Vector.h use memcpy/memset without including <cstring>
// themselves; in the full solver build some earlier translation unit
// always supplies it first, but that's not guaranteed for every
// consumer of this header, so include it explicitly here.
#include <cstring>

#include "Array.h"

#include <string>
#include <vector>

// Forward declared, not included: only field_reduction.cpp's Simulation-
// dependent methods need the full type. Keeping it out of this header
// avoids pulling Simulation.h's dependency graph into every translation
// unit that includes field_reduction.h.
class Simulation;
class ReductionConfig;

namespace field_reduction {

class Accumulator
{
  public:
    /// @brief Per-node running-statistic rows, one set per channel.
    enum StateRow { SUM = 0, MINV = 1, MAXV = 2, N = 3, NUM_STATE_ROWS = 4 };

    Accumulator();
    ~Accumulator();

    // Not copyable (owns heap-allocated exprtk pImpls); not needed.
    Accumulator(const Accumulator&) = delete;
    Accumulator& operator=(const Accumulator&) = delete;

    /// @brief Dependency-free setup: parses update_expr/finalize_expr
    /// exactly once per channel (parsing is the expensive part; per-node/
    /// per-timestep evaluation is then cheap) and allocates per-node,
    /// per-channel state for nNodes nodes / nChannels channels. Safe to
    /// call standalone, with no Simulation/mesh/MPI context -- this is
    /// what a unit test exercises directly.
    /// @throws std::runtime_error if either expression fails to parse.
    void init_core(int nNodes, int nChannels, const std::string& update_expr,
        const std::string& finalize_expr);

    /// @brief Per-node, per-channel update: copies this node/channel's
    /// running state into that channel's expression's bound scratch
    /// variables, evaluates update_expr (whose ":=" assignments mutate
    /// those scratch variables in place), then copies the mutated values
    /// back into state_. O(1), no re-parsing.
    void update(int local_node_idx, int channel, double value);

    /// @brief Evaluates finalize_expr once per node per channel,
    /// populating output().
    void finalize();

    /// @brief Zeros all per-node, per-channel state (and output). Used at
    /// the start of a fresh accumulation window.
    void reset();

    int n_nodes() const { return nNodes_; }
    int n_channels() const { return nChannels_; }
    double output(int channel, int local_node_idx) const { return output_(channel, local_node_idx); }
    double state(StateRow row, int channel, int local_node_idx) const { return channelState_[channel](row, local_node_idx); }

    /// @brief OSI-specific combine step, run after finalize(): given the 4
    /// already-finalized channels (0-2: componentwise WSS time-mean, 3: WSS
    /// magnitude time-mean -- both are plain "val_sum/n" means, fed by
    /// field_reduction_sim.cpp's update_from_solution() from one shared
    /// post::bpost call), computes the standard Oscillatory Shear Index
    /// per node: OSI = 0.5 * (1 - |mean_vec| / mean_mag), where mean_vec =
    /// (channel0, channel1, channel2) and mean_mag = channel3. A node with
    /// (near-)zero mean_mag has no measurable shear, so OSI is undefined
    /// there and reported as 0 rather than dividing by ~0. Populates
    /// combined_output(). Pure per-node math, no Simulation dependency --
    /// callable standalone (after init_core()+update()+finalize() with
    /// n_channels()==4) for unit testing.
    /// @throws std::runtime_error if n_channels() != 4.
    void combine_osi();

    /// @brief The single-channel result of combine_osi() (or, later,
    /// combine_transwss()) -- a scalar per node, independent of however
    /// many internal accumulation channels were used to compute it.
    double combined_output(int local_node_idx) const { return combinedOutput_(0, local_node_idx); }

    /// @brief Simulation-dependent setup for this invocation's accumulation
    /// window: validates the config (field/scope/mode combination, required
    /// fields), resolves Face_name to a mesh (WSS) or Mesh_name/the sole
    /// mesh (Velocity/Pressure), computes this invocation's target
    /// stopTS/window-start independently of main.cpp's own locals (see
    /// main.cpp:100-150 vs. the unrelated STOP_SIM-file local shadowing
    /// "stopTS" at main.cpp:559), determines nChannels, and calls
    /// init_core(). Then, only when continuing a previous simulation
    /// (com_mod.stFileFlag) and this invocation's windowStartCTS matches a
    /// valid <stFileName>_<name>_reduction.bin sidecar's own stored
    /// windowStartCTS (i.e. resuming a crash *inside* the same still-open
    /// window, not starting a new one), loads that sidecar's saved state_
    /// over the freshly-zeroed one -- see load_restart_sidecar(). No-op
    /// (state stays zeroed) for every other case: fresh start, no sidecar
    /// found, or a sidecar whose window doesn't match (the normal case: a
    /// new FSG coupling invocation opening a new window).
    void init(Simulation* simulation, const ReductionConfig& config);

    /// @brief Per-timestep hook, meant to be called once per converged
    /// timestep (after the Newton loop, alongside the existing per-step
    /// outputs). Once cTS has entered the trailing accumulation window,
    /// extracts this config's field (via post::bpost for face/WSS, or
    /// directly from com_mod.Yn for volume/Velocity|Pressure -- the latter
    /// needs no MPI collective at all, unlike bpost) and folds each
    /// channel's value into the running per-node state. No-op if not yet
    /// in-window.
    void update_from_solution(Simulation* simulation);

    /// @brief End-of-invocation hook: finalizes per-node, per-channel
    /// output, gathers it (and node coordinates) to rank 0 via the
    /// existing all_fun::global helper, and writes a single .vtu point
    /// cloud (0-based "GlobalNodeID" + "<Field>_reduction" point-data
    /// arrays, the latter with nChannels components; no connectivity
    /// beyond one degenerate vertex cell per point, since downstream
    /// consumption only needs the keyed values, and a .vtu with zero
    /// cells trips up common readers like meshio).
    void finalize_and_write(Simulation* simulation);

    /// @brief Restart-checkpoint hook, meant to be called alongside the
    /// existing output::write_restart() (same l1||l2 save cadence,
    /// main.cpp 594-596). Writes state_ (all channels) to
    /// <stFileName>_<name>_reduction.bin, one fixed-length record per rank
    /// at a rank-specific offset within the same shared file -- mirrors
    /// output::write_restart()/init_from_bin()'s own per-rank-record
    /// convention exactly, so it needs no MPI gather: every rank already
    /// owns a stable partition, so it can write/read its own slice
    /// directly.
    void write_restart_sidecar(Simulation* simulation);

  private:
    bool load_restart_sidecar(Simulation* simulation);

    int nNodes_ = 0;
    int nChannels_ = 0;
    std::vector<Array<double>> channelState_;  // nChannels_ x (NUM_STATE_ROWS, nNodes_)
    Array<double> output_;                      // (nChannels_, nNodes_)
    Array<double> combinedOutput_;               // (1, nNodes_); OSI/TransWSS only, see combine_osi()

    // exprtk plumbing is pImpl'd so exprtk.hpp (a ~47k-line template
    // header) is only ever included by field_reduction.cpp, not by every
    // file that includes this header. One ExprtkState per channel: same
    // expression *text* compiled independently per channel, since each
    // needs its own persistent scratch variables/symbol table.
    struct ExprtkState;
    std::vector<ExprtkState*> channelExpr_;

    std::string name_;
    std::string field_;          // "WSS" | "Velocity" | "Pressure" | "OSI"
    std::string mode_;           // "magnitude" | "componentwise"
    std::string outputFilePath_;
    int iM_ = -1;
    int stopTS_ = 0;
    int windowStartCTS_ = 0;
};

/// @brief Constructs simulation->fieldReducers from com_mod.reductions and
/// calls init() on each. No-op if com_mod.reductions is empty.
void init_all(Simulation* simulation);

/// @brief Calls update_from_solution() on every accumulator in
/// simulation->fieldReducers.
void update_all(Simulation* simulation);

/// @brief Calls finalize_and_write() on every accumulator in
/// simulation->fieldReducers.
void finalize_and_write_all(Simulation* simulation);

/// @brief Calls write_restart_sidecar() on every accumulator in
/// simulation->fieldReducers.
void write_restart_all(Simulation* simulation);

} // namespace field_reduction

#endif
