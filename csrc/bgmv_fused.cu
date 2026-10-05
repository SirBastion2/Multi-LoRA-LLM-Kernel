// Placeholder for fused QKV / gate-up shrink and fused shrink+expand (optimization K6–K7).
// Host entry points will be added when kernels are implemented.

#include <torch/extension.h>

void bgmv_fused_placeholder() {
  TORCH_CHECK(false, "bgmv_fused: not implemented yet");
}
