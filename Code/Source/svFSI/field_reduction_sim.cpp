// Simulation-dependent parts of field_reduction::Accumulator: config
// validation/resolution, per-timestep field extraction (post::bpost for
// face/WSS, com_mod.Yn directly for volume/Velocity|Pressure), VTU output,
// and the restart sidecar. Deliberately kept out of field_reduction.cpp
// (the dependency-free core -- init_core/update/finalize/reset), which
// test_field_reduction_standalone.cpp links against directly without
// pulling in Simulation.h/post.h/VtkData.h/mpi.h and their own heavy
// (VTK/PETSc/MPI) transitive dependencies.

#include "field_reduction.h"

#include "Simulation.h"
#include "all_fun.h"
#include "post.h"
#include "VtkData.h"

#include "mpi.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace field_reduction {

namespace {

// Fixed, arbitrary 4-byte tag identifying a field_reduction restart sidecar
// file, so a corrupt/foreign file at the same path is rejected (falls back
// to a fresh window) instead of being misread as valid state.
constexpr int kSidecarMagic = 0x46524231; // "FRB1", ASCII-ish
// Bumped to 2 for the added bufferSteps field (TransWSS's raw per-timestep
// buffer) -- the version-mismatch check below safely rejects any sidecar
// written by the previous (smaller) header layout, falling back to a
// fresh window exactly like any other invalidation case.
constexpr int kSidecarVersion = 2;

// One rank's fixed-length record in <stFileName>_<name>_reduction.bin, at
// offset (rank_id * recLn) -- recLn is the MAX of every rank's own record
// size (via MPI_Allreduce), exactly mirroring output::write_restart() /
// initialize::init_from_bin()'s own per-rank-record convention
// (output.cpp:233-285, initialize.cpp:118-121), so no MPI gather/scatter
// of the actual per-node state is ever needed: each rank already owns a
// stable mesh partition, so it can write/read its own slice directly.
struct SidecarHeader
{
  int magic = 0;
  int version = 0;
  int windowStartCTS = 0;
  int nNodes = 0;
  int nChannels = 0;
  // 0 for every field except TransWSS, where it equals Cycle_steps -- the
  // raw per-timestep WSS buffer (wssBuffer_) written/read right after
  // channelState_ when nonzero. See Accumulator::combine_transwss().
  int bufferSteps = 0;
};

// OSI (and, later, TransWSS) don't take a user Update_expr/Finalize_expr --
// their formula is fixed and combined across channels at finalize() (see
// Accumulator::combine_osi()), not expressible as one independent
// per-channel exprtk pair. Internally they still use the plain streaming
// mean (same text already used for Velocity/Pressure's own "time_average"
// convention) to accumulate each of their internal channels.
constexpr const char* kMeanUpdateExpr = "n := n + 1; val_sum := val_sum + val;";
constexpr const char* kMeanFinalizeExpr = "val_sum / n";

} // namespace

void Accumulator::init(Simulation* simulation, const ReductionConfig& config)
{
  auto& com_mod = simulation->com_mod;

  name_ = config.name;
  field_ = config.field;
  mode_ = config.mode;
  outputFilePath_ = config.outputFilePath;

  const std::string err_prefix = "field_reduction::Accumulator (Add_reduction name='" + name_ + "'): ";
  const bool isOsi = (field_ == "OSI");
  const bool isTransWss = (field_ == "TransWSS");
  const bool isFixedFormula = isOsi || isTransWss;

  if (field_ != "WSS" && field_ != "Velocity" && field_ != "Pressure" && !isFixedFormula) {
    throw std::runtime_error(err_prefix + "Field must be one of WSS|Velocity|Pressure|OSI|TransWSS, got '" + field_ + "'.");
  }
  if (config.scope != "face" && config.scope != "volume") {
    throw std::runtime_error(err_prefix + "Scope must be face|volume, got '" + config.scope + "'.");
  }
  if ((field_ == "WSS" || isFixedFormula) && config.scope != "face") {
    throw std::runtime_error(err_prefix + "Field=" + field_ + " requires Scope=face (post::bpost only computes WSS on a face).");
  }
  if (field_ != "WSS" && !isFixedFormula && config.scope != "volume") {
    throw std::runtime_error(err_prefix + "Field=" + field_ + " requires Scope=volume (only WSS/OSI/TransWSS support Scope=face).");
  }
  if (config.cycleSteps <= 0) {
    throw std::runtime_error(err_prefix + "Cycle_steps must be > 0 (got " + std::to_string(config.cycleSteps) + ").");
  }

  if (isFixedFormula) {
    // OSI/TransWSS's formula is fixed -- Reduction_mode/Update_expr/
    // Finalize_expr would silently be ignored if supplied, which is more
    // likely to hide a user mistake than help, so they're forbidden
    // outright.
    if (!config.mode.empty() || !config.updateExpr.empty() || !config.finalizeExpr.empty()) {
      throw std::runtime_error(err_prefix + "Field=" + field_ + " has a fixed formula and does not "
          "take Reduction_mode/Update_expr/Finalize_expr -- omit them.");
    }
  } else {
    if (mode_ != "magnitude" && mode_ != "componentwise") {
      throw std::runtime_error(err_prefix + "Reduction_mode must be magnitude|componentwise, got '" + mode_ + "'.");
    }
    if (field_ == "Pressure" && mode_ != "componentwise") {
      throw std::runtime_error(err_prefix + "Field=Pressure requires Reduction_mode=componentwise "
          "(magnitude of a scalar field is not a meaningful reduction here).");
    }
    if (config.updateExpr.empty() || config.finalizeExpr.empty()) {
      throw std::runtime_error(err_prefix + "Update_expr/Finalize_expr must both be non-empty.");
    }
  }

  const int nsd = com_mod.nsd;
  int nChannels = 1;

  if (field_ == "WSS" || isFixedFormula) {
    if (config.faceName.empty()) {
      throw std::runtime_error(err_prefix + "Field=" + field_ + " (Scope=face) requires a non-empty Face_name.");
    }
    int iFa = -1;
    all_fun::find_face(com_mod.msh, config.faceName, iM_, iFa);
    if (iFa != 0) {
      throw std::runtime_error(err_prefix + "Face_name '" + config.faceName + "' resolves to face index "
          + std::to_string(iFa) + " of its mesh, but post::bpost only computes WSS on face index 0.");
    }
    // OSI always accumulates 3 componentwise channels + 1 magnitude
    // channel internally, combined into 1 output value at finalize() via
    // combine_osi(). TransWSS accumulates just the 3 componentwise
    // channels (its own combine_transwss() also needs the raw per-
    // timestep buffer, allocated separately below via init_core()'s
    // bufferSteps/vecDim).
    nChannels = isOsi ? (nsd + 1) : (isTransWss ? nsd : ((mode_ == "magnitude") ? 1 : nsd));

  } else {
    // Velocity | Pressure, Scope=volume.
    if (!config.meshName.empty()) {
      all_fun::find_msh(com_mod.msh, config.meshName, iM_);
      if (iM_ < 0) {
        throw std::runtime_error(err_prefix + "Mesh_name '" + config.meshName + "' does not match any Add_mesh.");
      }
    } else if (com_mod.nMsh == 1) {
      iM_ = 0;
    } else {
      throw std::runtime_error(err_prefix + "Scope=volume with more than one mesh requires an explicit Mesh_name.");
    }
    if (field_ == "Velocity") {
      nChannels = (mode_ == "magnitude") ? 1 : nsd;
    } else {
      nChannels = 1; // Pressure: always scalar, already validated componentwise above.
    }
  }

  const auto& msh = com_mod.msh[iM_];
  init_core(msh.nNo, nChannels,
      isFixedFormula ? kMeanUpdateExpr : config.updateExpr,
      isFixedFormula ? kMeanFinalizeExpr : config.finalizeExpr,
      isTransWss ? config.cycleSteps : 0,
      isTransWss ? nsd : 0);

  if (isTransWss) {
    // Build a mesh-local-indexed per-node normal array from the resolved
    // face's own fa.nV (already computed by the solver -- see
    // post.cpp:155-170 for the identical face-local -> global-tnNo ->
    // mesh-local pattern this mirrors). Zero everywhere except the face's
    // own nodes, exactly like WSS's own tmpV (post::bpost fills the whole
    // mesh, meaningful only on the face) -- computed once here, not
    // per-timestep, since the geometry doesn't move within one fluid
    // solve invocation.
    faceNormal_.resize(nsd, msh.nNo);
    faceNormal_ = 0.0;
    const auto& fa = msh.fa[0];
    for (int a = 0; a < fa.nNo; a++) {
      int Ac = fa.gN(a);
      int localA = msh.lN(Ac);
      for (int i = 0; i < nsd; i++) {
        faceNormal_(i, localA) = fa.nV(i, a);
      }
    }
  }

  // This invocation's target final cTS, computed independently of
  // main.cpp's own local of the same name -- see main.cpp:100,148-150 (the
  // real one, in scope for the whole function) vs. the unrelated
  // STOP_SIM-file local that shadows it starting at main.cpp:559. Mirrors
  // main.cpp's own "int minTS = cTS + newTS; nTS = std::max(nTS, minTS);"
  // computed once, before the outer loop begins.
  stopTS_ = std::max(com_mod.nTS, com_mod.cTS + com_mod.newTS);
  windowStartCTS_ = stopTS_ - config.cycleSteps + 1;

  // Reset-vs-continue rule: a fresh start (stFileFlag false, including the
  // case where <Continue_previous_simulation> was true but initialize()
  // found no restart bin to actually load from, which resets the flag
  // itself -- see initialize.cpp:692-698) always keeps the zeroed state
  // init_core() just set up. Only a genuine continuation attempts to load
  // the sidecar, and even then only a window-matching sidecar is used (see
  // load_restart_sidecar()); anything else silently falls back to zeroed
  // state, which is exactly the correct behavior for a new window.
  if (com_mod.stFileFlag) {
    load_restart_sidecar(simulation);
  }
}

bool Accumulator::load_restart_sidecar(Simulation* simulation)
{
  auto& com_mod = simulation->com_mod;
  auto& cm = com_mod.cm;
  auto& cm_mod = simulation->cm_mod;

  std::string sidecarPath = com_mod.stFileName + "_" + name_ + "_reduction.bin";

  int channelBytes = 0;
  for (const auto& state : channelState_) {
    channelBytes += state.msize();
  }
  int bufferBytes = 0;
  for (const auto& step : wssBuffer_) {
    bufferBytes += step.msize();
  }
  int mySize = static_cast<int>(sizeof(SidecarHeader)) + channelBytes + bufferBytes;
  int recLn = 0;
  MPI_Allreduce(&mySize, &recLn, 1, cm_mod::mpint, MPI_MAX, cm.com());

  std::ifstream in(sidecarPath, std::ios::binary | std::ios::in);
  if (!in) {
    return false;
  }

  int processId = cm.tF(cm_mod);
  std::streampos readPos = static_cast<std::streampos>(processId - 1) * recLn;
  in.seekg(readPos);

  SidecarHeader onDisk;
  in.read(reinterpret_cast<char*>(&onDisk), sizeof(SidecarHeader));
  if (!in || onDisk.magic != kSidecarMagic || onDisk.version != kSidecarVersion) {
    return false;
  }
  // A mismatched node/channel/buffer count means the mesh/partition/config
  // changed since this sidecar was written -- unsafe to trust its
  // per-node state, which is implicitly keyed by local node index.
  if (onDisk.nNodes != nNodes_ || onDisk.nChannels != nChannels_ || onDisk.bufferSteps != bufferSteps_) {
    return false;
  }
  // Anything other than an exact window match means this sidecar is from
  // a different (earlier) accumulation window -- e.g. the normal case of
  // a new FSG coupling invocation opening a new window, not a
  // crash-recovery resume inside the same one.
  if (onDisk.windowStartCTS != windowStartCTS_) {
    return false;
  }

  for (auto& state : channelState_) {
    in.read(reinterpret_cast<char*>(state.data()), state.msize());
    if (!in) {
      return false;
    }
  }
  for (auto& step : wssBuffer_) {
    in.read(reinterpret_cast<char*>(step.data()), step.msize());
    if (!in) {
      return false;
    }
  }

  return true;
}

void Accumulator::write_restart_sidecar(Simulation* simulation)
{
  auto& com_mod = simulation->com_mod;
  auto& cm = com_mod.cm;
  auto& cm_mod = simulation->cm_mod;

  std::string sidecarPath = com_mod.stFileName + "_" + name_ + "_reduction.bin";

  SidecarHeader hdr;
  hdr.magic = kSidecarMagic;
  hdr.version = kSidecarVersion;
  hdr.windowStartCTS = windowStartCTS_;
  hdr.nNodes = nNodes_;
  hdr.nChannels = nChannels_;
  hdr.bufferSteps = bufferSteps_;

  int channelBytes = 0;
  for (const auto& state : channelState_) {
    channelBytes += state.msize();
  }
  int bufferBytes = 0;
  for (const auto& step : wssBuffer_) {
    bufferBytes += step.msize();
  }
  int mySize = static_cast<int>(sizeof(SidecarHeader)) + channelBytes + bufferBytes;
  int recLn = 0;
  MPI_Allreduce(&mySize, &recLn, 1, cm_mod::mpint, MPI_MAX, cm.com());

  // Master creates/truncates the shared file first, then every rank
  // (including master) reopens it and writes its own record at its own
  // offset -- mirrors output::write_restart()'s own create-then-block-
  // then-each-rank-writes pattern exactly (output.cpp:270-285).
  if (cm.mas(cm_mod)) {
    std::ofstream trunc(sidecarPath, std::ios::out | std::ios::binary);
    trunc.close();
  }

  // Blocks every rank until master's truncate above has completed, so no
  // rank races ahead and reopens the file before it exists.
  int fid = 0;
  cm.bcast(cm_mod, &fid);

  std::fstream out(sidecarPath, std::ios::in | std::ios::out | std::ios::binary);
  int processId = cm.tF(cm_mod);
  std::streampos writePos = static_cast<std::streampos>(processId - 1) * recLn;
  out.seekp(writePos);
  out.write(reinterpret_cast<const char*>(&hdr), sizeof(SidecarHeader));
  for (const auto& state : channelState_) {
    out.write(reinterpret_cast<const char*>(state.data()), state.msize());
  }
  for (const auto& step : wssBuffer_) {
    out.write(reinterpret_cast<const char*>(step.data()), step.msize());
  }
  out.close();
}

void Accumulator::update_from_solution(Simulation* simulation)
{
  auto& com_mod = simulation->com_mod;

  if (com_mod.cTS < windowStartCTS_) {
    return;
  }

  auto& msh = com_mod.msh[iM_];
  const int nsd = com_mod.nsd;

  if (field_ == "WSS" || field_ == "OSI" || field_ == "TransWSS") {
    Array<double> tmpV(consts::maxNSD, msh.nNo);
    post::bpost(simulation, msh, tmpV, com_mod.Yn, com_mod.Dn, consts::OutputType::outGrp_WSS);

    for (int a = 0; a < msh.nNo; a++) {
      if (field_ == "OSI") {
        // Feed both representations from this one bpost call: channels
        // 0..nsd-1 are the componentwise WSS vector, channel nsd is its
        // magnitude -- combine_osi() combines all 4 at finalize().
        double mag2 = 0.0;
        for (int i = 0; i < nsd; i++) {
          update(a, i, tmpV(i,a));
          mag2 += tmpV(i,a) * tmpV(i,a);
        }
        update(a, nsd, std::sqrt(mag2));
      } else if (field_ == "TransWSS") {
        // Channels 0..nsd-1 give mean_vec at finalize() (same as OSI's own
        // first 3 channels); the raw per-timestep vector is also buffered
        // here (record_raw()), since combine_transwss() needs each
        // individual timestep's WSS, not just their running sum -- the
        // transverse reference direction isn't known until the whole
        // window has been seen.
        int stepIdx = com_mod.cTS - windowStartCTS_;
        for (int i = 0; i < nsd; i++) {
          update(a, i, tmpV(i,a));
          record_raw(stepIdx, i, a, tmpV(i,a));
        }
      } else if (mode_ == "magnitude") {
        double mag2 = 0.0;
        for (int i = 0; i < nsd; i++) {
          mag2 += tmpV(i,a) * tmpV(i,a);
        }
        update(a, 0, std::sqrt(mag2));
      } else {
        for (int i = 0; i < nsd; i++) {
          update(a, i, tmpV(i,a));
        }
      }
    }

  } else {
    // Velocity | Pressure, Scope=volume: read directly from com_mod.Yn,
    // no post::bpost, no MPI collective at all (unlike WSS/face).
    // eq[0] mirrors post::bpost's own hardcoded "int iEq = 0;"
    // (post.cpp:138) -- appropriate for this project's single fluid
    // equation.
    const auto& eq = com_mod.eq[0];
    const int velOffset = eq.s;
    const int pressRow = eq.s + nsd;

    for (int a = 0; a < msh.nNo; a++) {
      int Ac = msh.gN(a);

      if (field_ == "Velocity") {
        if (mode_ == "magnitude") {
          double mag2 = 0.0;
          for (int i = 0; i < nsd; i++) {
            double v = com_mod.Yn(velOffset + i, Ac);
            mag2 += v * v;
          }
          update(a, 0, std::sqrt(mag2));
        } else {
          for (int i = 0; i < nsd; i++) {
            update(a, i, com_mod.Yn(velOffset + i, Ac));
          }
        }
      } else {
        // Pressure: always 1 channel, componentwise (validated in init()).
        update(a, 0, com_mod.Yn(pressRow, Ac));
      }
    }
  }
}

void Accumulator::finalize_and_write(Simulation* simulation)
{
  finalize();
  if (field_ == "OSI") {
    combine_osi();
  } else if (field_ == "TransWSS") {
    combine_transwss(faceNormal_);
  }

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
  // output_'s row count (nChannels_) passes through global() unchanged --
  // no reshaping needed for multi-channel output. OSI (and, later,
  // TransWSS) write their single combined channel instead of the raw
  // internal accumulation channels.
  const bool isCombined = (field_ == "OSI" || field_ == "TransWSS");
  Array<double> gx = all_fun::global(com_mod, cm_mod, msh, x);
  Array<double> gOut = all_fun::global(com_mod, cm_mod, msh, isCombined ? combinedOutput_ : output_);

  if (cm.slv(cm_mod)) {
    return;
  }

  const int gnNo = msh.gnNo;
  // VtkVtpData only implements set_point_data() for Vector<int> (used for
  // face id arrays like GlobalNodeID/GlobalElementID); its Array<double>
  // overload is an unimplemented stub that throws. VtkVtuData is the
  // reverse: Array<double>/Array<int> are implemented, Vector<int> is the
  // stub. Since the reduction is a double field, the output must be a
  // .vtu, and GlobalNodeID must go through the Array<int> overload, not
  // Vector<int>.
  Array<int> globalNodeId(1, gnNo);
  for (int a = 0; a < gnNo; a++) {
    globalNodeId(0,a) = a;
  }

  // One degenerate 1-node (VTK_VERTEX) cell per point. Not needed for
  // set_points()/set_point_data() themselves, but a VTU with zero cells
  // trips up common readers (confirmed: meshio's appended-data decoder
  // throws on an empty cell block) even though VTK's own writer accepts
  // it -- and downstream Python consumption needs this file to be
  // ordinarily readable, not just structurally legal.
  Array<int> vertexConn(1, gnNo);
  for (int a = 0; a < gnNo; a++) {
    vertexConn(0,a) = a;
  }

  auto vtk_writer = VtkData::create_writer(outputFilePath_);
  vtk_writer->set_points(gx);
  vtk_writer->set_connectivity(nsd, vertexConn);
  vtk_writer->set_point_data("GlobalNodeID", globalNodeId);
  // Array name is always "<Field>_reduction", independent of the block's
  // own name= attribute, so Python-side consumption always knows what to
  // look for regardless of how the user names their <Add_reduction> block.
  vtk_writer->set_point_data(field_ + "_reduction", gOut);
  vtk_writer->write();
  delete vtk_writer;
}

void init_all(Simulation* simulation)
{
  auto& com_mod = simulation->com_mod;
  for (const auto& config : com_mod.reductions) {
    auto* acc = new Accumulator();
    acc->init(simulation, config);
    simulation->fieldReducers.push_back(acc);
  }
}

void update_all(Simulation* simulation)
{
  for (auto* acc : simulation->fieldReducers) {
    acc->update_from_solution(simulation);
  }
}

void finalize_and_write_all(Simulation* simulation)
{
  for (auto* acc : simulation->fieldReducers) {
    acc->finalize_and_write(simulation);
  }
}

void write_restart_all(Simulation* simulation)
{
  for (auto* acc : simulation->fieldReducers) {
    acc->write_restart_sidecar(simulation);
  }
}

} // namespace field_reduction
