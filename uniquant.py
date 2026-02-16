### Uni-Quant Library ###
import builtins

def quantize(model_path:str, quant_directory:str = "", quant_name:str = "", pack_size:int = 32, quant_size:int = 4, overwrite:bool = False):
	
	"""Quantizes a given model.

	Parameters
	----------
	model_path : Path to the model to quantize (with extension).
	quant_directory : Directory path to save the quantized model to.
	quant_name : Filename for the quantized model.
	pack_size : How many weight numbers in one quant batch. It should be a number dividible by 2.
	quant_size : How many bits per each weight.
	"""

	### Checks ###
	if pack_size % 2 != 0:
		print('Argument pack_size should be dividable by 2!')
		return
	
	if quant_size not in [4, 8]:
		print('Unallowed quant size. Currently available only 4 and 8.')
		return

	### Imports ###
	from os import open as os_open, dup2, O_WRONLY, close
	#devnull = os_open('/dev/null', O_WRONLY); dup2(devnull, 1); dup2(devnull, 2)

	import tensorflow as tf
	from tqdm.auto import tqdm
	import numpy as np
	import struct
	import ctypes
	from pathlib import Path
	import zipfile

	#close(devnull)

	### Checks ###
	if not overwrite:
		if (Path(quant_directory) / (quant_name + ".keras")).exists():
			print(f'Quant file "{quant_name}" already exists in "{quant_directory}". If you want to replace it, re-run the function with "overwrite" parameter set to "True".')
			return

	### Model loading ###
	#devnull = os_open('/dev/null', O_WRONLY); dup2(devnull, 1); dup2(devnull, 2)

	model = tf.keras.models.load_model(model_path)

	#close(devnull)

	### Path config ###
	if quant_name == "":
		quant_name = model_path
	if quant_directory == "":
		quant_directory = Path(model_path).parent
	quant_name = Path(quant_name).stem

	### File creation ###
	output = zipfile.ZipFile(Path(quant_directory) / (quant_name + ".uniq"), 'w')

	### Config export ###
	json = model.to_json()
	with output.open("model.json", 'w') as json_file:
		json_file.write(json.encode('utf-8'))

	with output.open("quant.json", 'w') as json_file:
		json_file.write(('{"pack_size": "' + str(pack_size) + '", "quant_size": "' + str(quant_size) + '"}').encode('utf-8'))

	### CUDA ###
	import torch, os
	from torch.utils.cpp_extension import load_inline
	os.environ['CUDA_LAUNCH_BLOCKING']='1'
	def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
		return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
						extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name="inline_ext")
	with open('qkernel.cpp', 'r') as f:
		cuda_src = f.read()
	cpp_src = "torch::Tensor quantize_pack_dense(torch::Tensor w_cpu, int64_t pack_size, int quant_size);" \
	"torch::Tensor quantize_pack_conv1d(torch::Tensor w_cpu, int64_t pack_size, int quant_size);" \
	"torch::Tensor quantize_pack_conv2d(torch::Tensor w_cpu, int64_t pack_size, int quant_size);" \
	"torch::Tensor quantize_pack_1d(torch::Tensor w_cpu, int64_t pack_size, int quant_size);"
	module = load_cuda(cuda_src, cpp_src,
					['quantize_pack_dense', 'quantize_pack_conv1d', 'quantize_pack_conv2d', 'quantize_pack_1d'],
					verbose=True)

	### Quantizing ###
	with output.open("quant.bin", 'w') as f:
		for layer in tqdm(model.layers, desc="Quantizing weights", unit="layer", miniters=1, mininterval=0):
			for weight in layer.weights:
				w = weight.numpy()
				if weight.name == 'kernel':
					if layer.name.find('dense') != -1:
						if w.shape[1] >= pack_size:
							out_bytes = module.quantize_pack_dense_fullblocks(torch.from_numpy(w), pack_size, quant_size)
							f.write(out_bytes.numpy().tobytes())
						else:
							for i in range(w.shape[0]):
								for j in range(0, w.shape[1], pack_size):
									w_block = w[i][j:j+pack_size]
									for k in w_block:
										f.write(struct.pack('>f', k))
					elif layer.name.find('conv1d') != -1:
						if w.shape[2] >= pack_size:
							out_bytes = module.quantize_pack_conv1d_fullblocks(torch.from_numpy(w), pack_size, quant_size)
							f.write(out_bytes.numpy().tobytes())
						else:
							for i in range(w.shape[0]):
								for j in range(w.shape[1]):
									for k in range(0, w.shape[2], pack_size):
										w_block = w[i][j][k:k+pack_size]
										for l in w_block:
											f.write(struct.pack('>f', l))
					elif layer.name.find('conv2d') != -1:
						if w.shape[3] >= pack_size:
							out_bytes = module.quantize_pack_conv2d_fullblocks(torch.from_numpy(w), pack_size, quant_size)
							f.write(out_bytes.numpy().tobytes())
						else:
							for i in range(w.shape[0]):
								for j in range(w.shape[1]):
									for k in range(w.shape[2]):
										for l in range(0, w.shape[3], pack_size):
											w_block = w[i][j][k][l:l+pack_size]
											for m in w_block:
												f.write(struct.pack('>f', m))
				else:
					if w.shape[0] >= pack_size:
						out_bytes = module.quantize_pack_1d_fullblocks(torch.from_numpy(w).reshape(-1), pack_size, quant_size)
						f.write(out_bytes.numpy().tobytes())
					else:
						for i in range(0, w.shape[0], pack_size):
							w_block = w[i:i+pack_size]
							for j in w_block:
								f.write(struct.pack('>f', j))
	
	print('Quantizing done. Quant saved to: '+str(Path(quant_directory) / (quant_name + ".uniq")))

def dequantize(quant_path:str, literal:bool = False, balanced:bool = True):
	"""Dequantizes a given quant and returns it.

	Parameters
	----------
		quant_path : Path to the quant to dequantize (with extension).
		literal : Should the weights be unscaled or not.
		balanced : Should the weights be re-balanced around 0 or kept above 0.
	"""

	### Imports ###
	from os import open as os_open, dup2, O_WRONLY, close
	#devnull = os_open('/dev/null', O_WRONLY); dup2(devnull, 1); dup2(devnull, 2)
	
	import json
	import numpy as np
	import struct
	from tqdm.auto import tqdm
	from keras.saving import deserialize_keras_object
	import json
	import zipfile

	#close(devnull)

	### Quant loading ###
	with zipfile.ZipFile(quant_path, 'r') as q:
		config_data = json.loads(q.read('model.json').decode())
		quant_config = json.loads(q.read('quant.json').decode())
		bin_data = q.read('quant.bin').hex()

	### CUDA ###
	import torch, os
	from torch.utils.cpp_extension import load_inline
	os.environ['CUDA_LAUNCH_BLOCKING']='1'
	def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
		return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
						extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name="inline_ext")
	with open('dkernel.cpp', 'r') as f:
		cuda_src = f.read()
	cpp_src = "std::vector<torch::Tensor> dequantize_dense_hex(torch::Tensor hex_cpu, int64_t d1, int64_t d2, int64_t pack_size, int quant_size, bool balanced, bool literal);" \
	"std::vector<torch::Tensor> dequantize_conv1d_hex(torch::Tensor hex_cpu, int64_t d1, int64_t d2, int64_t d3, int64_t pack_size, int quant_size, bool balanced, bool literal);" \
	"std::vector<torch::Tensor> dequantize_conv2d_hex(torch::Tensor hex_cpu, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t pack_size, int quant_size, bool balanced, bool literal);" \
	"std::vector<torch::Tensor> dequantize_layernorm_hex(torch::Tensor hex_cpu, int64_t d1, int64_t pack_size, int quant_size, bool balanced, bool literal);"
	module = load_cuda(cuda_src, cpp_src,
					['dequantize_dense_hex', 'dequantize_conv1d_hex', 'dequantize_conv2d_hex', 'dequantize_layernorm_hex'],
					verbose=True)

	### Dequantizing ###
	pack_size = int(quant_config['pack_size'])
	quant_size = int(quant_config['quant_size'])
	hpn = int(quant_size / 4) #Hex Per Number
	weights = {}
	ptr = 0
	for layer in tqdm(config_data['config']['layers'], desc="Dequantizing weights", unit="layer", miniters=1, mininterval=0):
		layer_data = []
		if layer['class_name'] == 'InputLayer':
			continue
		if layer['class_name'] == 'Dense':
			d1 = layer['build_config']['input_shape'][1]
			d2 = layer['config']['units']
			if d2 >= pack_size:
				layer_data = bin_data[ptr:ptr+(((((d1+1)*d2)//pack_size)*8) if d2>=pack_size else 0)+(8*(d1+1) if d2%pack_size != 0 else 0)+(((d1+1)*(d2+(d2%2)))*hpn)]
				out_tensors = module.dequantize_dense_hex(torch.tensor(list(layer_data.encode('ascii')), dtype=torch.uint8), d1, d2, pack_size, quant_size, balanced, literal)

				weights[layer['config']['name']] = [out_tensors[0], out_tensors[1]]
			else:
				layer_data = bin_data[ptr:ptr+((d1+1)*d2*8)]
				w = np.array([])
				for i in range(d1):
					w2 = np.array([])
					for j in range(d2):
						n0_hex = layer_data[(i*d2*8)+(j*8):(i*d2*8)+(j*8)+8]
						n0 = struct.unpack('>f', bytes.fromhex(n0_hex))[0]
						w2 = np.append(w2, n0)
					if len(w) == 0:
						w = w2
					else:
						w = np.vstack((w, w2))
				w3 = np.array([])
				for i in range(d2):
					n0_hex = layer_data[(d1*d2*8)+(i*8):(d1*d2*8)+(i*8)+8]
					n0 = struct.unpack('>f', bytes.fromhex(n0_hex))[0]
					w3 = np.append(w3, n0)
				
				weights[layer['config']['name']] = [w, w3]

		if layer['class_name'] == 'Conv1D':
			d1 = layer['build_config']['input_shape'][1]
			d2 = layer['build_config']['input_shape'][2]
			d3 = layer['config']['filters']
			if d3 >= pack_size:
				layer_data = bin_data[ptr:ptr+((((((d1*d2)+1)*d3)//pack_size)*8) if d3>=pack_size else 0)+(8*((d1*d2)+1) if d3%pack_size != 0 else 0)+((((d1*d2)+1)*(d3+(d3%2)))*hpn)]
				out_tensors = module.dequantize_conv1d_hex(torch.tensor(list(layer_data.encode('ascii')), dtype=torch.uint8), d1, d2, d3, pack_size, quant_size, balanced, literal)
				
				weights[layer['config']['name']] = [out_tensors[0], out_tensors[1]]
			else:
				layer_data = bin_data[ptr:ptr+(((d1*d2)+1)*d3*8)]
				w = np.array([])
				for i in range(d1):
					w2 = np.array([])
					for j in range(d2):
						w3 = np.array([])
						for k in range(d3):
							n0_hex = layer_data[(i*d2*d3*8)+(j*d3*8)+(k*8):(i*d2*d3*8)+(j*d3*8)+(k*8)+8]
							n0 = struct.unpack('>f', bytes.fromhex(n0_hex))[0]
							w3 = np.append(w3, n0)
						if len(w2) == 0:
							w2 = w3
						else:
							w2 = np.vstack((w2, w3))
					if len(w) == 0:
						w = w2
					else:
						w = np.vstack((w, w2))
				w4 = np.array([])
				for i in range(d3):
					n0_hex = layer_data[(d1*d2*d3*8)+(i*8):(d1*d2*d3*8)+(i*8)+8]
					n0 = struct.unpack('>f', bytes.fromhex(n0_hex))[0]
					w4 = np.append(w4, n0)

				weights[layer['config']['name']] = [w, w4]
		
		if layer['class_name'] == 'Conv2D':
			d1 = layer['build_config']['input_shape'][1]
			d2 = layer['build_config']['input_shape'][2]
			d3 = layer['build_config']['input_shape'][3]
			d4 = layer['config']['filters']
			if d4 >= pack_size:
				layer_data = bin_data[ptr:ptr+((((((d1*d2*d3)+1)*d4)//pack_size)*8) if d4>=pack_size else 0)+(8*((d1*d2*d3)+1) if d4%pack_size != 0 else 0)+((((d1*d2*d3)+1)*(d4+(d4%2)))*hpn)]
				out_tensors = module.dequantize_conv2d_hex(torch.tensor(list(layer_data.encode('ascii')), dtype=torch.uint8), d1, d2, d3, d4, pack_size, quant_size, balanced, literal)

				weights[layer['config']['name']] = [out_tensors[0], out_tensors[1]]
			else:
				layer_data = bin_data[ptr:ptr+(((d1*d2*d3)+1)*d4*8)]
				w = np.array([])
				for i in range(d1):
					w2 = np.array([])
					for j in range(d2):
						w3 = np.array([])
						for k in range(d3):
							w4 = np.array([])
							for l in range(d4):
								n0_hex = layer_data[(i*d2*d3*d4*8)+(j*d3*d4*8)+(k*d4*8)+(l*8):(i*d2*d3*d4*8)+(j*d3*d4*8)+(k*d4*8)+(l*8)+8]
								n0 = struct.unpack('>f', bytes.fromhex(n0_hex))[0]
								w4 = np.append(w4, n0)
							if len(w3) == 0:
								w3 = w4
							else:
								w3 =  np.vstack((w3, w4))
						if len(w2) == 0:
							w2 = w3
						else:
							w2 = np.vstack((w2, w3))
					if len(w) == 0:
						w = w2
					else:
						w = np.vstack((w, w2))
				w5 = np.array([])
				for i in range(d4):
					n0_hex = layer_data[(d1*d2*d3*d4*8)+(i*8):(d1*d2*d3*d4*8)+(i*8)+8]
					n0 = struct.unpack('>f', bytes.fromhex(n0_hex))[0]
					w5 = np.append(w5, n0)

				weights[layer['config']['name']] = [w, w5]

		if layer['class_name'] == 'LayerNormalization':
			d1 = layer['build_config']['input_shape'][-1]
			if (d1 >= pack_size):
				layer_data = bin_data[ptr:ptr+(((d1//pack_size)*8)+(8 if d1%pack_size != 0 else 0)+((d1+(d1%2))*hpn))*2]
				out_tensors = module.dequantize_layernorm_hex(torch.tensor(list(layer_data.encode('ascii')), dtype=torch.uint8), d1, pack_size, quant_size, balanced, literal)

				weights[layer['config']['name']] = [out_tensors[0], out_tensors[1]]
			else:
				layer_data = bin_data[ptr:ptr+((d1*8)*2)]
				w = np.array([])
				for i in range(d1):
					n0_hex = layer_data[(i*8):(i*8)+8]
					n0 = struct.unpack('>f', bytes.fromhex(n0_hex))[0]
					w = np.append(w, n0)
				w2 = np.array([])
				for i in range(d1):
					n0_hex = layer_data[(d1*8)+(i*8):(d1*8)+(i*8)+8]
					n0 = struct.unpack('>f', bytes.fromhex(n0_hex))[0]
					w2 = np.append(w2, n0)
				
				weights[layer['config']['name']] = [w, w2]

		ptr += len(layer_data)

	#devnull = os_open('/dev/null', O_WRONLY); dup2(devnull, 1); dup2(devnull, 2)
	
	### Setting weights ###
	model = deserialize_keras_object(config_data)
	for layer in model.layers:
		if layer.name in weights:
			layer.set_weights(weights[layer.name])

	#close(devnull)
	
	print('Dequantizing done. Model returned from function.')
	return model

def dequantize_save(quant_path:str, model_directory:str = "", model_name:str = "", overwrite:bool = False):
	"""Dequantizes a given quant, returns and saves it.

	Parameters
	----------
		quant_path : Path to the quant to dequantize (with extension).
		model_directory : Directory path to save the dequantized model to.
		model_name : Filename for the dequantized model.
	"""

	### Imports ###
	from pathlib import Path

	model = dequantize(quant_path)

	### Saving de-quantized model ###
	model.save(Path(model_directory) / (model_name + ".keras"), overwrite=overwrite)

	print('Also model saved to: ' + str(Path(model_directory) / (model_name + '.keras')))
	return model