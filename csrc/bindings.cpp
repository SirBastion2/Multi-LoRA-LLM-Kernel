#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

void bgmv_shrink_cuda(const at::Tensor& x,
                      const at::Tensor& a_pool,
                      const at::Tensor& slots,
                      at::Tensor& v);

void bgmv_expand_cuda(const at::Tensor& v,
                      const at::Tensor& b_pool,
                      const at::Tensor& slots,
                      at::Tensor& y);

namespace persona {
void gemv_fp16_cuda(const at::Tensor& x, const at::Tensor& W, at::Tensor& y);
}

TORCH_LIBRARY(persona, m) {
  m.def("bgmv_shrink(Tensor x, Tensor a_pool, Tensor slots, Tensor(a!) v) -> ()");
  m.def("bgmv_expand(Tensor v, Tensor b_pool, Tensor slots, Tensor(a!) y) -> ()");
  m.def("gemv_fp16(Tensor x, Tensor W, Tensor(a!) y) -> ()");
}

TORCH_LIBRARY_IMPL(persona, CUDA, m) {
  m.impl("bgmv_shrink", &bgmv_shrink_cuda);
  m.impl("bgmv_expand", &bgmv_expand_cuda);
  m.impl("gemv_fp16", &persona::gemv_fp16_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("noop", []() { return 0; });
}
