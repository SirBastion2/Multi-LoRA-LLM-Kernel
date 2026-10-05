// CPU reference + optional GPU smoke test for standalone BGMV shrink.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "bgmv_standalone.cuh"
#include "device_buffer.hpp"

static void shrink_ref_cpu(const std::vector<half>& x,
                           const std::vector<half>& a_pool,
                           std::vector<float>& v,
                           const std::vector<int>& slots,
                           int T,
                           int K,
                           int R,
                           int S) {
  for (int t = 0; t < T; ++t) {
    int slot = slots[t];
    for (int r = 0; r < R; ++r) {
      if (slot < 0) {
        v[t * R + r] = 0.f;
        continue;
      }
      float acc = 0.f;
      for (int k = 0; k < K; ++k) {
        float xv = __half2float(x[t * K + k]);
        float av = __half2float(a_pool[(slot * R + r) * K + k]);
        acc += av * xv;
      }
      v[t * R + r] = acc;
    }
  }
}

int main() {
  const int T = 4;
  const int K = 128;
  const int R = 16;
  const int S = 2;
  std::vector<half> x(T * K);
  std::vector<half> a(S * R * K);
  std::vector<int> slots = {0, 1, -1, 0};
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = __float2half(0.01f * (i % 17));
  }
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = __float2half(0.001f * (i % 11));
  }

  std::vector<float> v_ref(T * R), v_gpu(T * R);
  shrink_ref_cpu(x, a, v_ref, slots, T, K, R, S);

  int dev = 0;
  if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
    cudaSetDevice(0);
  }
  DeviceBuffer d_x(T * K * sizeof(half));
  DeviceBuffer d_a(S * R * K * sizeof(half));
  DeviceBuffer d_slots(T * sizeof(int));
  DeviceBuffer d_v(T * R * sizeof(float));

  cudaMemcpy(d_x.data(), x.data(), d_x.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a.data(), a.data(), d_a.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_slots.data(), slots.data(), d_slots.size(), cudaMemcpyHostToDevice);

  constexpr int WARPS = 4;
  dim3 grid(T, (R + WARPS - 1) / WARPS);
  dim3 block(32, WARPS);
  standalone::bgmv_shrink_kernel<half, WARPS><<<grid, block>>>(
      reinterpret_cast<const half*>(d_x.data()),
      reinterpret_cast<const half*>(d_a.data()),
      reinterpret_cast<const int*>(d_slots.data()),
      reinterpret_cast<float*>(d_v.data()),
      K,
      R);
  cudaDeviceSynchronize();
  cudaMemcpy(v_gpu.data(), d_v.data(), d_v.size(), cudaMemcpyDeviceToHost);

  float max_err = 0.f;
  for (size_t i = 0; i < v_ref.size(); ++i) {
    max_err = fmaxf(max_err, fabsf(v_ref[i] - v_gpu[i]));
  }
  printf("standalone test_bgmv max_abs_err=%f\n", max_err);
  return max_err > 1e-2f ? 1 : 0;
}
