// Standalone Phase-2 sanity test for wss_reduction::Accumulator, exercised
// with no Simulation/mesh/MPI context -- just the exprtk-driven per-node
// update/finalize logic, checked against hand-computed reference values.
//
// Build (from Code/Source/svFSI/):
//   g++ -std=c++17 -include cstring -I. -I../../ThirdParty/exprtk \
//       test_wss_reduction_standalone.cpp wss_reduction.cpp Array.cpp \
//       Vector.cpp utils.cpp -o /tmp/test_wss_reduction
// (-include cstring works around Vector.h/utils.cpp using memcpy/memset
//  without including <cstring> themselves -- harmless in the real CMake
//  build, where some other header always supplies it first, but needed
//  when compiling this handful of files standalone.)
// Run:
//   /tmp/test_wss_reduction

#include "wss_reduction.h"

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

// Two synthetic wall nodes, three timesteps each -- the same shape as the
// last-cycle window the real accumulator sees, just tiny.
//   node 0: 1.0, 3.0, 2.0  -> TAWSS(mean) = 2.0,  amplitude(max-min) = 2.0
//   node 1: 0.5, 0.5, 0.5  -> TAWSS(mean) = 0.5,  amplitude(max-min) = 0.0
const double kNode0[3] = {1.0, 3.0, 2.0};
const double kNode1[3] = {0.5, 0.5, 0.5};

void test_tawss()
{
  std::printf("test_tawss (magnitude / TAWSS: mean_k(wss_mag))\n");

  wss_reduction::Accumulator acc;
  acc.init_core(2, "n := n + 1; wss_sum := wss_sum + wss_mag;", "wss_sum / n");

  for (int k = 0; k < 3; k++) {
    acc.update(0, kNode0[k]);
    acc.update(1, kNode1[k]);
  }
  acc.finalize();

  check("node 0 TAWSS", acc.output(0), 2.0);
  check("node 1 TAWSS", acc.output(1), 0.5);
}

void test_amplitude()
{
  std::printf("test_amplitude (peak-to-peak: max_k(wss_mag) - min_k(wss_mag))\n");

  wss_reduction::Accumulator acc;
  acc.init_core(2,
      "n := n + 1; "
      "minv := (n == 1) ? wss_mag : min(minv, wss_mag); "
      "maxv := (n == 1) ? wss_mag : max(maxv, wss_mag);",
      "maxv - minv");

  for (int k = 0; k < 3; k++) {
    acc.update(0, kNode0[k]);
    acc.update(1, kNode1[k]);
  }
  acc.finalize();

  check("node 0 amplitude", acc.output(0), 2.0);
  check("node 1 amplitude", acc.output(1), 0.0);
}

void test_reset_between_windows()
{
  std::printf("test_reset_between_windows (fresh window must not see stale state)\n");

  wss_reduction::Accumulator acc;
  acc.init_core(1, "n := n + 1; wss_sum := wss_sum + wss_mag;", "wss_sum / n");

  // First "window": accumulate then finalize.
  acc.update(0, 10.0);
  acc.update(0, 20.0);
  acc.finalize();
  check("window 1 TAWSS", acc.output(0), 15.0);

  // Reset (as init()/main.cpp would do when a NEW invocation's window
  // starts, per the restart sidecar's windowStartCTS mismatch case) and
  // accumulate a completely different window.
  acc.reset();
  acc.update(0, 1.0);
  acc.finalize();
  check("window 2 TAWSS (post-reset)", acc.output(0), 1.0);
}

void test_parse_error_throws()
{
  std::printf("test_parse_error_throws (bad expression must fail loudly, not silently)\n");

  wss_reduction::Accumulator acc;
  bool threw = false;
  try {
    acc.init_core(1, "n := n + 1; sum := sum + )(garbage", "sum / n");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check("threw on malformed update_expr", threw ? 1.0 : 0.0, 1.0);
}

} // namespace

int main()
{
  test_tawss();
  test_amplitude();
  test_reset_between_windows();
  test_parse_error_throws();

  if (failures == 0) {
    std::printf("\nALL TESTS PASSED\n");
    return 0;
  } else {
    std::printf("\n%d TEST(S) FAILED\n", failures);
    return 1;
  }
}
