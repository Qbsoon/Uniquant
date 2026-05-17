Uni-Quant
========

Small library to quantize/dequantize TensorFlow models using PyTorch CUDA kernels.

Notes
- This package compiles CUDA kernels at runtime using `torch.utils.cpp_extension.load_inline`.
- Installing and using the CUDA compilation requires a compatible CUDA toolkit on the target machine. (Tested with >=13)