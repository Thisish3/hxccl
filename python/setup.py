import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = os.path.dirname(os.path.abspath(__file__))
hx = os.path.join(root, "..")
setup(
    name="hxccl_torch",
    packages=["hxccl_torch"],
    ext_modules=[CUDAExtension(
        "hxccl_torch._C",
        ["hxccl_torch/csrc/backend.cpp"],
        include_dirs=[os.path.join(hx, "include"), "/usr/lib/x86_64-linux-gnu/openmpi/include"],
        extra_objects=[os.path.join(hx, "build-cuda", "libhxccl.a")],
        # mpi.h pulls in OpenMPI's C++ bindings when included from C++ (libhxccl.a and here)
        libraries=["mpi_cxx", "mpi", "cudart", "cuda"],  # cuda: driver API in libhxccl.a
        library_dirs=["/usr/lib/x86_64-linux-gnu/openmpi/lib", "/usr/local/cuda/lib64/stubs"],
    )],
    cmdclass={"build_ext": BuildExtension},
)
