#include <torch/types.h>
#include <c10/cuda/CUDAException.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_F32(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT_CPU_F32(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x); CHECK_F32(x)

static inline int64_t cdiv_i64(int64_t a, int64_t b) { return (a + b - 1) / b; }

__device__ __forceinline__ uint32_t bswap32(uint32_t v) {
	return ((v & 0x000000FFu) << 24) | ((v & 0x0000FF00u) << 8) | ((v & 0x00FF0000u) >> 8) | ((v & 0xFF000000u) >> 24);
}

__device__ __forceinline__ void write_be_f32(uint8_t* dst, float v) {
	uint32_t u = __float_as_uint(v);
	u = bswap32(u);
	dst[0] = (uint8_t)(u & 0xFF);
	dst[1] = (uint8_t)((u >> 8) & 0xFF);
	dst[2] = (uint8_t)((u >> 16) & 0xFF);
	dst[3] = (uint8_t)((u >> 24) & 0xFF);
}

__device__ __forceinline__ int clamp_int(int v, int lo, int hi) {
	return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void quant_pack_kernel(
	int mode,
	const float* w,
	int64_t d0, int64_t d1, int64_t d2, int64_t d3,
	int64_t pack_size,
	int quant_size,
	int half_point,
	int64_t blocks_per_inner,
	int64_t total_blocks,
	int64_t stride,
	int64_t rem,
	int64_t rem_payload,
	int64_t rem_stride,
	int64_t row_bytes,
	uint8_t* out
) {
	int64_t bid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (bid >= total_blocks) return;

	int64_t row_idx = bid / blocks_per_inner;
	int64_t blk_in_row = bid % blocks_per_inner;
	int is_rem_block = (rem > 0 && blk_in_row == (blocks_per_inner - (rem > 0 ? 1 : 0))) ? 1 : 0;
	int64_t actual_pack = is_rem_block ? rem : pack_size;

	int64_t base = 0;

	if (mode == 0) {
		base = bid * pack_size;
	} else if (mode == 1) {
		int64_t i = bid / blocks_per_inner;
		int64_t jb = bid - i * blocks_per_inner;
		int64_t start = jb * pack_size;
		base = i * d1 + start;
	} else if (mode == 2) {
		int64_t tmp = bid;
		int64_t kb = tmp % blocks_per_inner;
		tmp /= blocks_per_inner;
		int64_t j = tmp % d1;
		int64_t i = tmp / d1;
		int64_t start = kb * pack_size;
		base = (i * d1 + j) * d2 + start;
	} else {
		int64_t tmp = bid;
		int64_t lb = tmp % blocks_per_inner;
		tmp /= blocks_per_inner;
		int64_t k = tmp % d2;
		tmp /= d2;
		int64_t j = tmp % d1;
		int64_t i = tmp / d1;
		int64_t start = lb * pack_size;
		base = (((i * d1 + j) * d2 + k) * d3) + start;
	}

	const float* src = w + base;
	int64_t out_off = row_idx * row_bytes;
	if (is_rem_block) {
		out_off += (blocks_per_inner - 1) * stride;
	} else {
		out_off += blk_in_row * stride;
	}
	uint8_t* dst = out + out_off;

	float max_abs = 0.0f;
	for (int64_t t = 0; t < actual_pack; ++t) {
		float a = fabsf(src[t]);
		if (a > max_abs) max_abs = a;
	}

	float scale = max_abs / (float)(half_point - 1);
	write_be_f32(dst, scale);
	dst += 4;

	int lo = -(half_point - 1);
	int hi = (half_point - 1);

	if (quant_size == 8) {
		for (int64_t t = 0; t < actual_pack; ++t) {
			float x = src[t];
			float qf;
			if (scale == 0.0f) qf = (x > 0.0f) ? (float)hi : ((x < 0.0f) ? (float)lo : 0.0f);
			else qf = nearbyintf(x / scale);
			int q = clamp_int((int)qf, lo, hi) + half_point;
			dst[t] = (uint8_t)q;
		}
	} else {
		int64_t o = 0;
		int64_t padded = actual_pack + (actual_pack & 1);
		for (int64_t t = 0; t < padded; t += 2) {
			float x0 = (t < actual_pack) ? src[t] : 0.0f;
			float qf0;
			if (scale == 0.0f) qf0 = (x0 > 0.0f) ? (float)hi : ((x0 < 0.0f) ? (float)lo : 0.0f);
			else qf0 = nearbyintf(x0 / scale);
			int q0 = clamp_int((int)qf0, lo, hi) + half_point;

			float x1 = (t + 1 < actual_pack) ? src[t + 1] : 0.0f;
			float qf1;
			if (scale == 0.0f) qf1 = (x1 > 0.0f) ? (float)hi : ((x1 < 0.0f) ? (float)lo : 0.0f);
			else qf1 = nearbyintf(x1 / scale);
			int q1 = clamp_int((int)qf1, lo, hi) + half_point;

			dst[o++] = (uint8_t)(((q0 & 0x0F) << 4) | (q1 & 0x0F));
		}
	}
}

static torch::Tensor quantize_pack_impl(int mode, torch::Tensor w_cpu, int64_t pack_size, int quant_size) {
	CHECK_INPUT_CPU_F32(w_cpu);
	TORCH_CHECK(pack_size > 0, "pack_size must be > 0");
	TORCH_CHECK(quant_size == 4 || quant_size == 8, "quant_size must be 4 or 8");
	if (quant_size == 4) TORCH_CHECK((pack_size & 1) == 0, "pack_size must be even for quant_size=4");

	int half_point = (1 << quant_size) / 2;

	int64_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
	int64_t blocks_per_inner = 0;
	int64_t total_blocks = 0;

	if (mode == 0) {
		TORCH_CHECK(w_cpu.dim() == 1, "w_cpu must be 1D");
		d0 = w_cpu.size(0);
		blocks_per_inner = d0 / pack_size;
		if (d0 % pack_size != 0) blocks_per_inner += 1;
		total_blocks = blocks_per_inner;
	} else if (mode == 1) {
		TORCH_CHECK(w_cpu.dim() == 2, "w_cpu must be 2D");
		d0 = w_cpu.size(0);
		d1 = w_cpu.size(1);
		blocks_per_inner = d1 / pack_size;
		if (d1 % pack_size != 0) blocks_per_inner += 1;
		total_blocks = d0 * blocks_per_inner;
	} else if (mode == 2) {
		TORCH_CHECK(w_cpu.dim() == 3, "w_cpu must be 3D");
		d0 = w_cpu.size(0);
		d1 = w_cpu.size(1);
		d2 = w_cpu.size(2);
		blocks_per_inner = d2 / pack_size;
		if (d2 % pack_size != 0) blocks_per_inner += 1;
		total_blocks = d0 * d1 * blocks_per_inner;
	} else {
		TORCH_CHECK(w_cpu.dim() == 4, "w_cpu must be 4D");
		d0 = w_cpu.size(0);
		d1 = w_cpu.size(1);
		d2 = w_cpu.size(2);
		d3 = w_cpu.size(3);
		blocks_per_inner = d3 / pack_size;
		if (d3 % pack_size != 0) blocks_per_inner += 1;
		total_blocks = d0 * d1 * d2 * blocks_per_inner;
	}

	int64_t payload = (quant_size == 4) ? (pack_size / 2) : pack_size;
	int64_t stride = 4 + payload;

	int64_t inner_dim = (mode == 0) ? d0 : ((mode == 1) ? d1 : ((mode == 2) ? d2 : d3));
	int64_t rem = inner_dim % pack_size;
	int64_t rem_payload = 0, rem_stride = 0;
	if (rem > 0) {
		int64_t rem_padded = rem + (rem & 1);
		rem_payload = (quant_size == 4) ? (rem_padded / 2) : rem;
		rem_stride = 4 + rem_payload;
	}

	int64_t num_rows = total_blocks / blocks_per_inner;
	int64_t full_blocks = inner_dim / pack_size;
	int64_t row_bytes = full_blocks * stride + (rem > 0 ? rem_stride : 0);
	int64_t total_bytes = num_rows * row_bytes;

	auto w = w_cpu.to(torch::kCUDA);
	auto out = torch::zeros({total_bytes}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8));

	int threads = 256;
	int grid = (int)cdiv_i64(total_blocks, threads);

	quant_pack_kernel<<<grid, threads>>>(
		mode,
		(const float*)w.data_ptr<float>(),
		d0, d1, d2, d3,
		pack_size,
		quant_size,
		half_point,
		blocks_per_inner,
		total_blocks,
		stride,
		rem,
		rem_payload,
		rem_stride,
		row_bytes,
		(uint8_t*)out.data_ptr<uint8_t>()
	);
	C10_CUDA_KERNEL_LAUNCH_CHECK();

	return out.to(torch::kCPU);
}

torch::Tensor quantize_pack_1d(torch::Tensor w_cpu, int64_t pack_size, int quant_size) {
	return quantize_pack_impl(0, w_cpu, pack_size, quant_size);
}

torch::Tensor quantize_pack_2d(torch::Tensor w_cpu, int64_t pack_size, int quant_size) {
	return quantize_pack_impl(1, w_cpu, pack_size, quant_size);
}

torch::Tensor quantize_pack_3d(torch::Tensor w_cpu, int64_t pack_size, int quant_size) {
	return quantize_pack_impl(2, w_cpu, pack_size, quant_size);
}

torch::Tensor quantize_pack_4d(torch::Tensor w_cpu, int64_t pack_size, int quant_size) {
	return quantize_pack_impl(3, w_cpu, pack_size, quant_size);
}