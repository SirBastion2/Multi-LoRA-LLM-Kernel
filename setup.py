from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CSRC = [
    "csrc/bindings.cpp",
    "csrc/bgmv_shrink.cu",
    "csrc/bgmv_expand.cu",
    "csrc/gemv_fp16.cu",
    "csrc/bgmv_fused.cu",
]

setup(
    name="persona-serve",
    version="0.1.0",
    packages=["persona"],
    ext_modules=[
        CUDAExtension(
            name="persona._C",
            sources=CSRC,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
