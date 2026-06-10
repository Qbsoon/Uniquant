Uni-Quant
========

Small library to quantize/dequantize TensorFlow models using PyTorch CUDA kernels.

## Requirements

- **Python**: 3.13.13 (haven't tested on any other)
- **CUDA Toolkit**: >=12.8
- **Python Dependencies**: All required packages are listed in `requirements.txt`

### Installing Dependencies

```bash
pip install -r requirements.txt
```

## Installation from pip

```bash
pip install uni-quant-cuda
```

## Usage

### Importing Functions

```python
from uniquant import quantize, dequantize, dequantize_save
```

### Main Functions

#### `quantize(model_path, quant_directory="", quant_name="", pack_size=32, quant_size=4, overwrite=False)`
Quantizes a TensorFlow or XGBoost model.

**Arguments:**
- `model_path` (str): Path to the model to quantize (with extension)
- `quant_directory` (str): Directory path to save the quantized model
- `quant_name` (str): Filename for the quantized model
- `pack_size` (int): Number of weights in one quantization batch (must be divisible by 2)
- `quant_size` (int): Number of bits per weight (available: 4 or 8)
- `overwrite` (bool): Whether to overwrite existing file

#### `dequantize(quant_path, literal=False, balanced=True)`
Dequantizes a model and returns it.

**Arguments:**
- `quant_path` (str): Path to the .uniq file to dequantize
- `literal` (bool): Whether weights should be unscaled
- `balanced` (bool): Whether weights should be balanced around 0

#### `dequantize_save(quant_path, model_directory="", model_name="", overwrite=False)`
Dequantizes a model, saves it, and returns it.

**Arguments:**
- `quant_path` (str): Path to the .uniq file to dequantize
- `model_directory` (str): Directory path to save the dequantized model
- `model_name` (str): Filename for the dequantized model
- `overwrite` (bool): Whether to overwrite existing file

## Notes

- This package compiles CUDA kernels at runtime using `torch.utils.cpp_extension.load_inline`.
- Installing and using the CUDA compilation requires a compatible CUDA toolkit on the target machine (tested with >=12.8).