#include <torch/types.h>
#include <c10/cuda/CUDAException.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_U8(x) TORCH_CHECK(x.scalar_type() == torch::kUInt8, #x " must be uint8")
#define CHECK_INPUT_CPU_HEX(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x); CHECK_U8(x)

static inline int64_t cdiv_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

__device__ __forceinline__ uint8_t hex_nibble(uint8_t c) {
	if (c >= '0' && c <= '9') return (uint8_t)(c - '0');
	if (c >= 'a' && c <= 'f') return (uint8_t)(10 + (c - 'a'));
	return (uint8_t)(10 + (c - 'A'));
}

__device__ __forceinline__ uint8_t hex_byte(const uint8_t* s2) {
	return (uint8_t)((hex_nibble(s2[0]) << 4) | hex_nibble(s2[1]));
}

__device__ __forceinline__ float read_be_f32_from_hex8(const uint8_t* s8) {
	uint32_t b0 = (uint32_t)hex_byte(s8 + 0);
	uint32_t b1 = (uint32_t)hex_byte(s8 + 2);
	uint32_t b2 = (uint32_t)hex_byte(s8 + 4);
	uint32_t b3 = (uint32_t)hex_byte(s8 + 6);
	uint32_t u = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
	return __uint_as_float(u);
}

__global__ void dequant_hex_kernel(
	int mode,
	const uint8_t* hex,
	float* out_a,
	float* out_b,
	int64_t d1, int64_t d2, int64_t d3, int64_t d4,
	int64_t pack_size,
	int quant_size,
	int half_point,
	int balanced,
	int literal,
	int64_t total_elems
) {
	int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total_elems) return;

	int64_t batches = 0;
	int64_t rem = 0;

	int64_t batch_shift = (int64_t)(pack_size * (quant_size / 4.0f));
	int64_t rec_full = 8 + batch_shift;

	int64_t rem_shift = 0;
	int64_t rec_rem = 0;

	int64_t elems_weights = 0;

	int64_t rows = 0;

	if (mode == 0 || mode == 4) {
		batches = d2 / pack_size;
		rem = d2 % pack_size;
		rem_shift = (int64_t)(rem * (quant_size / 4.0f));
		rec_rem = rem ? (8 + rem_shift) : 0;
		rows = d1;
		elems_weights = d1 * d2;
	} else if (mode == 1) {
		batches = d3 / pack_size;
		rem = d3 % pack_size;
		rem_shift = (int64_t)(rem * (quant_size / 4.0f));
		rec_rem = rem ? (8 + rem_shift) : 0;
		rows = d1 * d2;
		elems_weights = d1 * d2 * d3;
	} else if (mode == 2) {
		batches = d4 / pack_size;
		rem = d4 % pack_size;
		rem_shift = (int64_t)(rem * (quant_size / 4.0f));
		rec_rem = rem ? (8 + rem_shift) : 0;
		rows = d1 * d2 * d3;
		elems_weights = d1 * d2 * d3 * d4;
	} else {
		batches = d1 / pack_size;
		rem = d1 % pack_size;
		rem_shift = (int64_t)(rem * (quant_size / 4.0f));
		rec_rem = rem ? (8 + rem_shift) : 0;
		rows = 1;
		elems_weights = d1;
	}

	int is_second = 0;
	int is_bias = 0;
	int64_t pos = idx;

	if (mode == 4) {
		is_bias = 0;
	} else if (mode == 3) {
		if (pos >= elems_weights) {
			is_second = 1;
			pos -= elems_weights;
		}
		is_bias = is_second;
	} else {
		if (pos >= elems_weights) {
			is_bias = 1;
			pos -= elems_weights;
		}
	}

	int64_t row = 0;
	int64_t col = 0;

	if (mode == 0 || mode == 4) {
		if (!is_bias) {
			row = pos / d2;
			col = pos - row * d2;
		} else {
			row = rows;
			col = pos;
		}
	} else if (mode == 1) {
		if (!is_bias) {
			row = pos / d3;
			col = pos - row * d3;
		} else {
			row = rows;
			col = pos;
		}
	} else if (mode == 2) {
		if (!is_bias) {
			row = pos / d4;
			col = pos - row * d4;
		} else {
			row = rows;
			col = pos;
		}
	} else {
		row = 0;
		col = pos;
	}

	int64_t row_hex = batches * rec_full + (rem ? rec_rem : 0);
	int64_t base_hex = 0;

	if (mode == 3) {
		base_hex = is_second ? (row_hex) : 0;
	} else {
		base_hex = row * row_hex;
		if (is_bias) base_hex = rows * row_hex;
	}

	int64_t blk = col / pack_size;
	int64_t within = col - blk * pack_size;

	int64_t rec_hex_off = 0;
	int64_t payload_hex_off = 0;
	float scale = 0.0f;

	if (blk < batches) {
		rec_hex_off = base_hex + blk * rec_full;
		scale = read_be_f32_from_hex8(hex + rec_hex_off);
		payload_hex_off = rec_hex_off + 8;
		if (quant_size == 8) {
			int q = (int)hex_byte(hex + payload_hex_off + within * 2);
			int n = q - (balanced ? half_point : 0);
			float v = literal ? (float)n : ((float)n * scale);
			out_a[idx] = v;
			if (mode == 3 || is_bias) out_b[pos] = v;
		} else {
			int64_t byte_i = within >> 1;
			uint8_t b = hex_byte(hex + payload_hex_off + byte_i * 2);
			int q = (within & 1) ? (int)(b & 0x0F) : (int)((b >> 4) & 0x0F);
			if ((within & 1) && q == 0) {
				out_a[idx] = 0.0f;
				if (mode == 3 || is_bias) out_b[pos] = 0.0f;
			} else {
				int n = q - (balanced ? half_point : 0);
				float v = literal ? (float)n : ((float)n * scale);
				out_a[idx] = v;
				if (mode == 3 || is_bias) out_b[pos] = v;
			}
		}
	} else {
		if (!rem) {
			out_a[idx] = 0.0f;
			if (mode == 3 || is_bias) out_b[pos] = 0.0f;
			return;
		}
		int64_t rblk = blk - batches;
		rec_hex_off = base_hex + batches * rec_full + rblk * rec_rem;
		scale = read_be_f32_from_hex8(hex + rec_hex_off);
		payload_hex_off = rec_hex_off + 8;

		if (quant_size == 8) {
			int q = (int)hex_byte(hex + payload_hex_off + within * 2);
			int n = q - (balanced ? half_point : 0);
			float v = literal ? (float)n : ((float)n * scale);
			out_a[idx] = v;
			if (mode == 3 || is_bias) out_b[pos] = v;
		} else {
			int64_t byte_i = within >> 1;
			uint8_t b = hex_byte(hex + payload_hex_off + byte_i * 2);
			int q = (within & 1) ? (int)(b & 0x0F) : (int)((b >> 4) & 0x0F);
			if ((within & 1) && q == 0) {
				out_a[idx] = 0.0f;
				if (mode == 3 || is_bias) out_b[pos] = 0.0f;
			} else {
				int n = q - (balanced ? half_point : 0);
				float v = literal ? (float)n : ((float)n * scale);
				out_a[idx] = v;
				if (mode == 3 || is_bias) out_b[pos] = v;
			}
		}
	}
}

static std::vector<torch::Tensor> dequantize_hex_impl(
	int mode,
	torch::Tensor hex_cpu,
	int64_t d1, int64_t d2, int64_t d3, int64_t d4,
	int64_t pack_size,
	int quant_size,
	bool balanced,
	bool literal
) {
	CHECK_INPUT_CPU_HEX(hex_cpu);
	TORCH_CHECK(pack_size > 0, "pack_size must be > 0");
	TORCH_CHECK(quant_size == 4 || quant_size == 8, "quant_size must be 4 or 8");
	int half_point = (1 << quant_size) / 2;

	int64_t total = 0;
	if (mode == 0) {
		TORCH_CHECK(d2 >= pack_size, "d2 must be >= pack_size");
		total = d1 * d2 + d2;
	} else if (mode == 1) {
		TORCH_CHECK(d3 >= pack_size, "d3 must be >= pack_size");
		total = d1 * d2 * d3 + d3;
	} else if (mode == 2) {
		TORCH_CHECK(d4 >= pack_size, "d4 must be >= pack_size");
		total = d1 * d2 * d3 * d4 + d4;
	} else if (mode == 4) {
		TORCH_CHECK(d2 >= pack_size, "d2 must be >= pack_size");
		total = d1 * d2;
	} else {
		TORCH_CHECK(d1 >= pack_size, "d1 must be >= pack_size");
		total = d1 + d1;
	}

	auto hex_gpu = hex_cpu.to(torch::kCUDA);
	auto out_a_gpu = torch::empty({total}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
	torch::Tensor out_b_gpu;

	if (mode == 0) out_b_gpu = torch::empty({d2}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
	else if (mode == 1) out_b_gpu = torch::empty({d3}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
	else if (mode == 2) out_b_gpu = torch::empty({d4}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
	else if (mode == 3) out_b_gpu = torch::empty({d1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
	else out_b_gpu = torch::empty({1}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

	int threads = 256;
	int grid = (int)cdiv_i64(total, threads);

	dequant_hex_kernel<<<grid, threads>>>(
		mode,
		(const uint8_t*)hex_gpu.data_ptr<uint8_t>(),
		(float*)out_a_gpu.data_ptr<float>(),
		(float*)out_b_gpu.data_ptr<float>(),
		d1, d2, d3, d4,
		pack_size,
		quant_size,
		half_point,
		(int)balanced,
		(int)literal,
		total
	);
	C10_CUDA_KERNEL_LAUNCH_CHECK();

	if (mode == 0) {
		auto w = out_a_gpu.narrow(0, 0, d1 * d2).view({d1, d2}).to(torch::kCPU);
		auto b = out_b_gpu.to(torch::kCPU);
		return { w, b };
	}
	if (mode == 1) {
		auto w = out_a_gpu.narrow(0, 0, d1 * d2 * d3).view({d1, d2, d3}).to(torch::kCPU);
		auto b = out_b_gpu.to(torch::kCPU);
		return { w, b };
	}
	if (mode == 2) {
		auto w = out_a_gpu.narrow(0, 0, d1 * d2 * d3 * d4).view({d1, d2, d3, d4}).to(torch::kCPU);
		auto b = out_b_gpu.to(torch::kCPU);
		return { w, b };
	}
	if (mode == 4) {
		return { out_a_gpu.to(torch::kCPU) };
	}
	auto a = out_a_gpu.narrow(0, 0, d1).to(torch::kCPU);
	auto b = out_b_gpu.to(torch::kCPU);
	return { a, b };
}

std::vector<torch::Tensor> dequantize_dense_hex(torch::Tensor hex_cpu, int64_t d1, int64_t d2, int64_t pack_size, int quant_size, bool balanced, bool literal) {
	return dequantize_hex_impl(0, hex_cpu, d1, d2, 0, 0, pack_size, quant_size, balanced, literal);
}

std::vector<torch::Tensor> dequantize_conv1d_hex(torch::Tensor hex_cpu, int64_t d1, int64_t d2, int64_t d3, int64_t pack_size, int quant_size, bool balanced, bool literal) {
	return dequantize_hex_impl(1, hex_cpu, d1, d2, d3, 0, pack_size, quant_size, balanced, literal);
}

std::vector<torch::Tensor> dequantize_conv2d_hex(torch::Tensor hex_cpu, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t pack_size, int quant_size, bool balanced, bool literal) {
	return dequantize_hex_impl(2, hex_cpu, d1, d2, d3, d4, pack_size, quant_size, balanced, literal);
}

std::vector<torch::Tensor> dequantize_gru_hex(torch::Tensor hex_cpu, int64_t d1, int64_t units, int64_t biases, int64_t pack_size, int quant_size, bool balanced, bool literal) {
	int64_t d2 = 3 * units;
	int64_t rows = d1 + units + biases;

	auto tmp = dequantize_hex_impl(4, hex_cpu, rows, d2, 0, 0, pack_size, quant_size, balanced, literal);
	auto flat = tmp[0];

	auto full = flat.view({rows, d2});
	auto w_in = full.narrow(0, 0, d1).contiguous();
	auto w_rec = full.narrow(0, d1, units).contiguous();
	auto b2d = full.narrow(0, d1 + units, biases).contiguous();
	if (biases == 1) {
		auto b1d = b2d.view({d2}).contiguous();
		return { w_in, w_rec, b1d };
	}
	return { w_in, w_rec, b2d };
}

std::vector<torch::Tensor> dequantize_layernorm_hex(torch::Tensor hex_cpu, int64_t d1, int64_t pack_size, int quant_size, bool balanced, bool literal) {
	return dequantize_hex_impl(3, hex_cpu, d1, 0, 0, 0, pack_size, quant_size, balanced, literal);
}