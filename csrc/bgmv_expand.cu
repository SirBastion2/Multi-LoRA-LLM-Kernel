#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "common.cuh"

namespace persona {

template <typename T, int R>
__global__ void bgmv_expand_kernel(const float* __restrict__ v,
                                   const T* __restrict__ b_pool,
                                   const int* __restrict__ slots,
                                   T* __restrict__ y,
                                   int N) {
  const int t = blockIdx.x;
  const int n = blockIdx.y * blockDim.x + threadIdx.x;
  const int slot = slots[t];
  if (slot < 0 || n >= N) {
    return;
  }

  const T* b_row = b_pool + ((size_t)slot * N + n) * R;
  float acc = 0.f;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    acc += v[t * R + r] * to_float<T>(b_row[r]);
  }
  const size_t idx = (size_t)t * N + n;
  y[idx] = from_float<T>(to_float<T>(y[idx]) + acc);
}

template <typename T, int R>
void launch_bgmv_expand(const float* v,
                        const T* b_pool,
                        const int* slots,
                        T* y,
                        int T_tok,
                        int N,
                        cudaStream_t stream) {
  constexpr int kBlock = 256;
  dim3 grid(T_tok, (N + kBlock - 1) / kBlock);
  dim3 block(kBlock);
  bgmv_expand_kernel<T, R><<<grid, block, 0, stream>>>(v, b_pool, slots, y, N);
}

template <typename T, int R>
void dispatch_rank(const at::Tensor& v,
                   const at::Tensor& b_pool,
                   const at::Tensor& slots,
                   at::Tensor& y) {
  const int T_tok = static_cast<int>(v.size(0));
  const int N = static_cast<int>(b_pool.size(1));
  launch_bgmv_expand<T, R>(v.data_ptr<float>(),
                             reinterpret_cast<const T*>(b_pool.data_ptr()),
                             slots.data_ptr<int>(),
                             reinterpret_cast<T*>(y.data_ptr()),
                             T_tok,
                             N,
                             at::cuda::getCurrentCUDAStream());
}

template <typename T>
void dispatch_bgmv_expand(const at::Tensor& v,
                          const at::Tensor& b_pool,
                          const at::Tensor& slots,
                          at::Tensor& y,
                          int64_t R) {
  switch (R) {
    case 8:
      dispatch_rank<T, 8>(v, b_pool, slots, y);
      break;
    case 16:
      dispatch_rank<T, 16>(v, b_pool, slots, y);
      break;
    case 32:
      dispatch_rank<T, 32>(v, b_pool, slots, y);
      break;
    case 64:
      dispatch_rank<T, 64>(v, b_pool, slots, y);
      break;
    default:
      TORCH_CHECK(false, "bgmv_expand: R must be 8, 16, 32, or 64");
  }
}

}  // namespace persona

void bgmv_expand_cuda(const at::Tensor& v,
                      const at::Tensor& b_pool,
                      const at::Tensor& slots,
                      at::Tensor& y) {
  TORCH_CHECK(v.is_cuda() && b_pool.is_cuda() && slots.is_cuda() && y.is_cuda(),
              "bgmv_expand: all tensors must be CUDA");
  TORCH_CHECK(v.is_contiguous() && b_pool.is_contiguous() && slots.is_contiguous() &&
                  y.is_contiguous(),
              "bgmv_expand: tensors must be contiguous");
  TORCH_CHECK(slots.scalar_type() == at::kInt, "bgmv_expand: slots must be int32");
  TORCH_CHECK(v.scalar_type() == at::kFloat, "bgmv_expand: v must be float32");

  const int64_t T_tok = v.size(0);
  const int64_t R = v.size(1);
  const int64_t S = b_pool.size(0);
  const int64_t N = b_pool.size(1);
  TORCH_CHECK(b_pool.size(2) == R, "bgmv_expand: b_pool rank dim mismatch");
  TORCH_CHECK(slots.size(0) == T_tok, "bgmv_expand: slots length must match T");
  TORCH_CHECK(y.size(0) == T_tok && y.size(1) == N, "bgmv_expand: y shape mismatch");
  TORCH_CHECK(N % 8 == 0, "bgmv_expand: N must be a multiple of 8");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(b_pool.data_ptr()) % 16 == 0,
              "bgmv_expand: b_pool must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0,
              "bgmv_expand: y must be 16-byte aligned");

  for (int64_t t = 0; t < T_tok; ++t) {
    int slot = slots[t].item<int>();
    TORCH_CHECK(slot >= -1 && slot < S, "bgmv_expand: slot out of range");
  }

  const at::cuda::OptionalCUDAGuard guard(v.device());

  if (y.scalar_type() == at::kHalf) {
    TORCH_CHECK(b_pool.scalar_type() == at::kHalf, "bgmv_expand: dtype mismatch");
    persona::dispatch_bgmv_expand<half>(v, b_pool, slots, y, R);
  } else if (y.scalar_type() == at::kBFloat16) {
    TORCH_CHECK(b_pool.scalar_type() == at::kBFloat16, "bgmv_expand: dtype mismatch");
    persona::dispatch_bgmv_expand<__nv_bfloat16>(v, b_pool, slots, y, R);
  } else {
    TORCH_CHECK(false, "bgmv_expand: only float16 and bfloat16 supported");
  }
}
