// Warm-up GEMV kernel (one warp per output row). Compared against cuBLAS in standalone/bench.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "common.cuh"

namespace persona {

template <typename T>
__global__ void gemv_fp16_kernel(const T* __restrict__ x,
                                 const T* __restrict__ W,
                                 T* __restrict__ y,
                                 int K,
                                 int N) {
  const int n = blockIdx.x;
  const int lane = threadIdx.x;
  if (n >= N) {
    return;
  }
  const T* w_row = W + (size_t)n * K;
  float acc = 0.f;
  for (int k = lane; k < K; k += 32) {
    acc += to_float<T>(w_row[k]) * to_float<T>(x[k]);
  }
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  if (lane == 0) {
    y[n] = from_float<T>(acc);
  }
}

void gemv_fp16_cuda(const at::Tensor& x, const at::Tensor& W, at::Tensor& y) {
  TORCH_CHECK(x.is_cuda() && W.is_cuda() && y.is_cuda(), "gemv_fp16: CUDA only");
  TORCH_CHECK(x.dim() == 1 && W.dim() == 2 && y.dim() == 1, "gemv_fp16: shape");
  const int K = static_cast<int>(x.size(0));
  const int N = static_cast<int>(W.size(0));
  TORCH_CHECK(W.size(1) == K && y.size(0) == N, "gemv_fp16: mismatch");

  if (x.scalar_type() == at::kHalf) {
    gemv_fp16_kernel<half><<<N, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(x.data_ptr()),
        reinterpret_cast<const half*>(W.data_ptr()),
        reinterpret_cast<half*>(y.data_ptr()),
        K,
        N);
  } else {
    TORCH_CHECK(false, "gemv_fp16: half only in v1 scaffold");
  }
}

}  // namespace persona
