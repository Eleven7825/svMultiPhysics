// Standalone sanity test for field_reduction::Accumulator, exercised with
// no Simulation/mesh/MPI context -- just the exprtk-driven per-node,
// per-channel update/finalize logic, checked against hand-computed
// reference values.
//
// Build (from Code/Source/svFSI/): only links field_reduction.cpp (the
// dependency-free core), NOT field_reduction_sim.cpp -- the latter needs
// Simulation.h/post.h/VtkData.h/mpi.h and their own (VTK/PETSc/MPI) heavy
// transitive dependencies, which this standalone test deliberately avoids.
//   g++ -std=c++17 -include cstring -I. -I../../ThirdParty/exprtk \
//       test_field_reduction_standalone.cpp field_reduction.cpp Array.cpp \
//       Vector.cpp utils.cpp -o /tmp/test_field_reduction
// (-include cstring works around Vector.h/utils.cpp using memcpy/memset
//  without including <cstring> themselves -- harmless in the real CMake
//  build, where some other header always supplies it first, but needed
//  when compiling this handful of files standalone.)
// Run:
//   /tmp/test_field_reduction

#include "field_reduction.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

bool close(double a, double b, double tol = 1e-12)
{
  return std::fabs(a - b) <= tol;
}

int failures = 0;

void check(const char* label, double got, double expected)
{
  if (close(got, expected)) {
    std::printf("  PASS  %-40s got=%.12g expected=%.12g\n", label, got, expected);
  } else {
    std::printf("  FAIL  %-40s got=%.12g expected=%.12g\n", label, got, expected);
    failures++;
  }
}

// Two synthetic nodes, three timesteps each -- the same shape as the
// last-cycle window the real accumulator sees, just tiny.
//   node 0: 1.0, 3.0, 2.0  -> mean = 2.0,  amplitude(max-min) = 2.0
//   node 1: 0.5, 0.5, 0.5  -> mean = 0.5,  amplitude(max-min) = 0.0
const double kNode0[3] = {1.0, 3.0, 2.0};
const double kNode1[3] = {0.5, 0.5, 0.5};

const char* kMeanUpdate = "n := n + 1; val_sum := val_sum + val;";
const char* kMeanFinalize = "val_sum / n";
const char* kAmplitudeUpdate =
    "n := n + 1; "
    "minv := (n == 1) ? val : min(minv, val); "
    "maxv := (n == 1) ? val : max(maxv, val);";
const char* kAmplitudeFinalize = "maxv - minv";

void test_mean_1channel()
{
  std::printf("test_mean_1channel (single-channel time average, e.g. TAWSS/Pressure)\n");

  field_reduction::Accumulator acc;
  acc.init_core(2, 1, kMeanUpdate, kMeanFinalize);

  for (int k = 0; k < 3; k++) {
    acc.update(0, 0, kNode0[k]);
    acc.update(1, 0, kNode1[k]);
  }
  acc.finalize();

  check("node 0 mean", acc.output(0, 0), 2.0);
  check("node 1 mean", acc.output(0, 1), 0.5);
}

void test_amplitude_1channel()
{
  std::printf("test_amplitude_1channel (single-channel peak-to-peak, e.g. WSS magnitude)\n");

  field_reduction::Accumulator acc;
  acc.init_core(2, 1, kAmplitudeUpdate, kAmplitudeFinalize);

  for (int k = 0; k < 3; k++) {
    acc.update(0, 0, kNode0[k]);
    acc.update(1, 0, kNode1[k]);
  }
  acc.finalize();

  check("node 0 amplitude", acc.output(0, 0), 2.0);
  check("node 1 amplitude", acc.output(0, 1), 0.0);
}

void test_componentwise_3channel()
{
  std::printf("test_componentwise_3channel (e.g. Velocity time-average, one independent "
      "reduction per component)\n");

  // Single node, 3 channels (x/y/z), each with its own 3-timestep signal,
  // same shape as kNode0/kNode1 above but reused per-channel with a
  // constant offset so each channel's expected mean is distinguishable.
  field_reduction::Accumulator acc;
  acc.init_core(1, 3, kMeanUpdate, kMeanFinalize);

  for (int k = 0; k < 3; k++) {
    acc.update(0, 0, kNode0[k]);          // channel 0 (x): mean -> 2.0
    acc.update(0, 1, kNode1[k]);          // channel 1 (y): mean -> 0.5
    acc.update(0, 2, kNode0[k] + 10.0);   // channel 2 (z): mean -> 12.0
  }
  acc.finalize();

  check("channel 0 (x) mean", acc.output(0, 0), 2.0);
  check("channel 1 (y) mean", acc.output(1, 0), 0.5);
  check("channel 2 (z) mean", acc.output(2, 0), 12.0);
}

void test_reset_between_windows()
{
  std::printf("test_reset_between_windows (fresh window must not see stale state)\n");

  field_reduction::Accumulator acc;
  acc.init_core(1, 1, kMeanUpdate, kMeanFinalize);

  // First "window": accumulate then finalize.
  acc.update(0, 0, 10.0);
  acc.update(0, 0, 20.0);
  acc.finalize();
  check("window 1 mean", acc.output(0, 0), 15.0);

  // Reset (as init()/main.cpp would do when a NEW invocation's window
  // starts, per the restart sidecar's windowStartCTS mismatch case) and
  // accumulate a completely different window.
  acc.reset();
  acc.update(0, 0, 1.0);
  acc.finalize();
  check("window 2 mean (post-reset)", acc.output(0, 0), 1.0);
}

void test_osi_combine()
{
  std::printf("test_osi_combine (OSI = 0.5*(1 - |mean_vec| / mean_mag), 4 internal channels)\n");

  // Two nodes, 4 channels each (x, y, z, magnitude), 2 synthetic timesteps.
  //   node 0: WSS oscillates between (3,0,0) and (-1,0,0) --
  //     mean_vec = (1,0,0), |mean_vec| = 1; mean_mag = (3+1)/2 = 2
  //     OSI = 0.5*(1 - 1/2) = 0.25
  //   node 1: WSS constant at (2,0,0) both steps (no oscillation) --
  //     mean_vec = (2,0,0), |mean_vec| = 2; mean_mag = 2
  //     OSI = 0.5*(1 - 2/2) = 0.0
  const double node0[2][3] = {{3.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}};
  const double node1[2][3] = {{2.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};

  field_reduction::Accumulator acc;
  acc.init_core(2, 4, kMeanUpdate, kMeanFinalize);

  for (int t = 0; t < 2; t++) {
    double mag0 = std::sqrt(node0[t][0]*node0[t][0] + node0[t][1]*node0[t][1] + node0[t][2]*node0[t][2]);
    double mag1 = std::sqrt(node1[t][0]*node1[t][0] + node1[t][1]*node1[t][1] + node1[t][2]*node1[t][2]);
    for (int i = 0; i < 3; i++) {
      acc.update(0, i, node0[t][i]);
      acc.update(1, i, node1[t][i]);
    }
    acc.update(0, 3, mag0);
    acc.update(1, 3, mag1);
  }
  acc.finalize();
  acc.combine_osi();

  check("node 0 OSI (oscillating WSS)", acc.combined_output(0), 0.25);
  check("node 1 OSI (constant WSS)", acc.combined_output(1), 0.0);
}

void test_osi_combine_wrong_channels_throws()
{
  std::printf("test_osi_combine_wrong_channels_throws (combine_osi requires exactly 4 channels)\n");

  field_reduction::Accumulator acc;
  acc.init_core(1, 3, kMeanUpdate, kMeanFinalize);
  acc.update(0, 0, 1.0);
  acc.update(0, 1, 1.0);
  acc.update(0, 2, 1.0);
  acc.finalize();

  bool threw = false;
  try {
    acc.combine_osi();
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check("threw on n_channels() != 4", threw ? 1.0 : 0.0, 1.0);
}

void test_parse_error_throws()
{
  std::printf("test_parse_error_throws (bad expression must fail loudly, not silently)\n");

  field_reduction::Accumulator acc;
  bool threw = false;
  try {
    acc.init_core(1, 1, "n := n + 1; sum := sum + )(garbage", "sum / n");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check("threw on malformed update_expr", threw ? 1.0 : 0.0, 1.0);
}

} // namespace

int main()
{
  test_mean_1channel();
  test_amplitude_1channel();
  test_componentwise_3channel();
  test_reset_between_windows();
  test_osi_combine();
  test_osi_combine_wrong_channels_throws();
  test_parse_error_throws();

  if (failures == 0) {
    std::printf("\nALL TESTS PASSED\n");
    return 0;
  } else {
    std::printf("\n%d TEST(S) FAILED\n", failures);
    return 1;
  }
}
