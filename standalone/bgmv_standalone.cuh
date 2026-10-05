#pragma once

// Torch-free BGMV kernels for the standalone harness (mirrors csrc/ v1).

#include <cuda_fp16.h>
#include <cstdint>

namespace standalone {

template <typename T>
__device__ inline float to_float(T x);

template <>
__device__ inline float to_float<half>(half x) {
  return __half2float(x);
}

template <typename T>
__device__ inline float dot8(uint4 av, uint4 xv) {
  float acc = 0.f;
  const T* a_ptr = reinterpret_cast<const T*>(&av);
  const T* x_ptr = reinterpret_cast<const T*>(&xv);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc += to_float(a_ptr[i]) * to_float(x_ptr[i]);
  }
  return acc;
}

template <typename T, int WARPS>
__global__ void bgmv_shrink_kernel(const T* x,
                                   const T* a_pool,
                                   const int* slots,
                                   float* v,
                                   int K,
                                   int R) {
  const int t = blockIdx.x;
  const int r = blockIdx.y * WARPS + threadIdx.y;
  const int lane = threadIdx.x;
  if (r >= R) {
    return;
  }
  const int slot = slots[t];
  if (slot < 0) {
    if (lane == 0) {
      v[t * R + r] = 0.f;
    }
    return;
  }
  const T* a_row = a_pool + ((size_t)slot * R + r) * K;
  const T* x_row = x + (size_t)t * K;
  float acc = 0.f;
  for (int k = lane * 8; k < K; k += 32 * 8) {
    uint4 av = *reinterpret_cast<const uint4*>(a_row + k);
    uint4 xv = *reinterpret_cast<const uint4*>(x_row + k);
    acc += dot8<T>(av, xv);
  }
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  if (lane == 0) {
    v[t * R + r] = acc;
  }
}

}  // namespace standalone
