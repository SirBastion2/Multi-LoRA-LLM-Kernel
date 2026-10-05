#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "common.cuh"

namespace persona {

template <typename T, int WARPS>
__global__ void bgmv_shrink_kernel(const T* __restrict__ x,
                                   const T* __restrict__ a_pool,
                                   const int* __restrict__ slots,
                                   float* __restrict__ v,
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

template <typename T, int WARPS>
void launch_bgmv_shrink(const T* x,
                        const T* a_pool,
                        const int* slots,
                        float* v,
                        int T_tok,
                        int K,
                        int R,
                        cudaStream_t stream) {
  dim3 grid(T_tok, (R + WARPS - 1) / WARPS);
  dim3 block(32, WARPS);
  bgmv_shrink_kernel<T, WARPS><<<grid, block, 0, stream>>>(x, a_pool, slots, v, K, R);
}

template <typename T>
void dispatch_bgmv_shrink(const at::Tensor& x,
                          const at::Tensor& a_pool,
                          const at::Tensor& slots,
                          at::Tensor& v) {
  const int T_tok = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int R = static_cast<int>(a_pool.size(1));
  const T* x_ptr = reinterpret_cast<const T*>(x.data_ptr());
  const T* a_ptr = reinterpret_cast<const T*>(a_pool.data_ptr());
  const int* slots_ptr = slots.data_ptr<int>();
  float* v_ptr = v.data_ptr<float>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int WARPS = 4;
  launch_bgmv_shrink<T, WARPS>(x_ptr, a_ptr, slots_ptr, v_ptr, T_tok, K, R, stream);
}

}  // namespace persona

void bgmv_shrink_cuda(const at::Tensor& x,
                      const at::Tensor& a_pool,
                      const at::Tensor& slots,
                      at::Tensor& v) {
  TORCH_CHECK(x.is_cuda() && a_pool.is_cuda() && slots.is_cuda() && v.is_cuda(),
              "bgmv_shrink: all tensors must be CUDA");
  TORCH_CHECK(x.is_contiguous() && a_pool.is_contiguous() && slots.is_contiguous() &&
                  v.is_contiguous(),
              "bgmv_shrink: tensors must be contiguous");
  TORCH_CHECK(slots.scalar_type() == at::kInt, "bgmv_shrink: slots must be int32");
  TORCH_CHECK(v.scalar_type() == at::kFloat, "bgmv_shrink: v must be float32");

  const int64_t T_tok = x.size(0);
  const int64_t K = x.size(1);
  const int64_t S = a_pool.size(0);
  const int64_t R = a_pool.size(1);
  TORCH_CHECK(a_pool.size(2) == K, "bgmv_shrink: a_pool shape mismatch");
  TORCH_CHECK(slots.size(0) == T_tok, "bgmv_shrink: slots length must match T");
  TORCH_CHECK(v.size(0) == T_tok && v.size(1) == R, "bgmv_shrink: v shape mismatch");
  TORCH_CHECK(K % 8 == 0, "bgmv_shrink: K must be a multiple of 8");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0,
              "bgmv_shrink: x must be 16-byte aligned");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a_pool.data_ptr()) % 16 == 0,
              "bgmv_shrink: a_pool must be 16-byte aligned");

  for (int64_t t = 0; t < T_tok; ++t) {
    int slot = slots[t].item<int>();
    TORCH_CHECK(slot >= -1 && slot < S, "bgmv_shrink: slot out of range");
  }

  const at::cuda::OptionalCUDAGuard guard(x.device());

  if (x.scalar_type() == at::kHalf) {
    TORCH_CHECK(a_pool.scalar_type() == at::kHalf, "bgmv_shrink: dtype mismatch");
    persona::dispatch_bgmv_shrink<half>(x, a_pool, slots, v);
  } else if (x.scalar_type() == at::kBFloat16) {
    TORCH_CHECK(a_pool.scalar_type() == at::kBFloat16, "bgmv_shrink: dtype mismatch");
    persona::dispatch_bgmv_shrink<__nv_bfloat16>(x, a_pool, slots, v);
  } else {
    TORCH_CHECK(false, "bgmv_shrink: only float16 and bfloat16 supported");
  }
}
