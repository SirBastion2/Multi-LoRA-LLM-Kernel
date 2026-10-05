#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <stdexcept>

#define PERSONA_CUDA_CHECK(call)                                                     \
  do {                                                                               \
    cudaError_t err = (call);                                                        \
    if (err != cudaSuccess) {                                                        \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    }                                                                                \
  } while (0)

namespace persona {

template <typename T>
__device__ inline float to_float(T x);

template <>
__device__ inline float to_float<half>(half x) {
  return __half2float(x);
}

template <>
__device__ inline float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

template <typename T>
__device__ inline T from_float(float x);

template <>
__device__ inline half from_float<half>(float x) {
  return __float2half(x);
}

template <>
__device__ inline __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

template <typename T>
__device__ inline float dot8(uint4 av, uint4 xv) {
  float acc = 0.f;
  const T* a_ptr = reinterpret_cast<const T*>(&av);
  const T* x_ptr = reinterpret_cast<const T*>(&xv);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    acc += to_float<T>(a_ptr[i]) * to_float<T>(x_ptr[i]);
  }
  return acc;
}

}  // namespace persona
