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

	### Quantizing ###
	half_point = int((2**quant_size)/2)
	progress = tqdm(total=len(model.layers), desc="Quantizing weights", unit="layer", miniters=1, mininterval=0)
	with output.open("quant.bin", 'w') as f:
		for layer in model.layers:
			for weight in layer.weights:
				w = weight.numpy()
				packed = 0
				if weight.name == 'kernel':
					if layer.name.find('dense') != -1:
						for i in range(w.shape[0]):
							for j in range(0, w.shape[1], pack_size):
								w_block = w[i][j:j+pack_size]
								if w.shape[1] >= pack_size:
									scale = (np.max(np.abs(w_block))) / (half_point - 1)
									w_block_q = np.clip(np.round(w_block / scale), -(half_point - 1), (half_point - 1)).astype(np.int16)
									w_block_q = w_block_q + half_point
									f.write(struct.pack('>f', scale))
									if (quant_size == 4):
										packed = [ctypes.c_uint8((w_block_q[k] & 0x0F) << 4 | (w_block_q[k+1] & 0x0F) if k+1 < len(w_block_q) else 0).value for k in range(0, len(w_block_q), 2)]
									elif (quant_size == 8):
										packed = [ctypes.c_uint8(k).value for k in range(0, len(w_block_q))]
									f.write(bytearray(packed))
								else:
									for k in w_block:
										f.write(struct.pack('>f', k))
					elif layer.name.find('conv1d') != -1:
						for i in range(w.shape[0]):
							for j in range(w.shape[1]):
								for k in range(0, w.shape[2], pack_size):
									w_block = w[i][j][k:k+pack_size]
									if w.shape[2] >= pack_size:
										scale = (np.max(np.abs(w_block))) / (half_point - 1)
										w_block_q = np.clip(np.round(w_block / scale), -(half_point - 1), (half_point - 1)).astype(np.int16)
										w_block_q = w_block_q + half_point
										f.write(struct.pack('>f', scale))
										if (quant_size == 4):
											packed = [ctypes.c_uint8((w_block_q[m] & 0x0F) << 4 | (w_block_q[m+1] & 0x0F) if m+1 < len(w_block_q) else 0).value for m in range(0, len(w_block_q), 2)]
										elif (quant_size == 8):
											packed = [ctypes.c_uint8(m).value for m in range(0, len(w_block_q))]
										f.write(bytearray(packed))
									else:
										for m in w_block:
											f.write(struct.pack('>f', m))
					elif layer.name.find('conv2d') != -1:
						for i in range(w.shape[0]):
							for j in range(w.shape[1]):
								for k in range(w.shape[2]):
									for l in range(0, w.shape[3], pack_size):
										w_block = w[i][j][k][l:l+pack_size]
										if w.shape[3] >= pack_size:
											scale = (np.max(np.abs(w_block))) / (half_point - 1)
											w_block_q = np.clip(np.round(w_block / scale), -(half_point - 1), (half_point - 1)).astype(np.int16)
											w_block_q = w_block_q + half_point
											f.write(struct.pack('>f', scale))
											if (quant_size == 4):
												packed = [ctypes.c_uint8((w_block_q[m] & 0x0F) << 4 | (w_block_q[m+1] & 0x0F) if m+1 < len(w_block_q) else 0).value for m in range(0, len(w_block_q), 2)]
											elif (quant_size == 8):
												packed = [ctypes.c_uint8(m).value for m in range(0, len(w_block_q))]
											f.write(bytearray(packed))
										else:
											for m in w_block:
												f.write(struct.pack('>f', m))
				else:
					for i in range(0, w.shape[0], pack_size):
						w_block = w[i:i+pack_size]
						if w.shape[0] >= pack_size:
							scale = (np.max(np.abs(w_block))) / (half_point - 1)
							w_block_q = np.clip(np.round(w_block / scale), -(half_point - 1), (half_point - 1)).astype(np.int16)
							w_block_q = w_block_q + half_point
							f.write(struct.pack('>f', scale))
							if (quant_size == 4):
								packed = [ctypes.c_uint8((w_block_q[k] & 0x0F) << 4 | (w_block_q[k+1] & 0x0F) if k+1 < len(w_block_q) else 0).value for k in range(0, len(w_block_q), 2)]
							elif (quant_size == 8):
								packed = [ctypes.c_uint8(k).value for k in range(0, len(w_block_q))]
							f.write(bytearray(packed))
						else:
							for k in w_block:
								f.write(struct.pack('>f', k))
			progress.update(1)
	
	print('Quantizing done. Quant saved to: '+str(Path(quant_directory) / (quant_name + ".uniq")))

def dequantize(quant_path:str):
	"""Dequantizes a given quant and returns it.

	Parameters
	----------
		quant_path : Path to the quant to dequantize (with extension).
	"""

	### Imports ###
	from os import open as os_open, dup2, O_WRONLY, close
	#devnull = os_open('/dev/null', O_WRONLY); dup2(devnull, 1); dup2(devnull, 2)
	
	import json
	import numpy as np
	import struct
	from tqdm.auto import tqdm
	from pathlib import Path
	from keras.saving import deserialize_keras_object
	import json
	import zipfile

	#close(devnull)

	### Quant loading ###
	with zipfile.ZipFile(quant_path, 'r') as q:
		config_data = json.loads(q.read('model.json').decode())
		quant_config = json.loads(q.read('quant.json').decode())
		bin_data = q.read('quant.bin').hex()

	### Dequantizing ###
	pack_size = int(quant_config['pack_size'])
	quant_size = int(quant_config['quant_size'])
	batch_shift = int(pack_size * (quant_size / 4))
	hpn = int(quant_size / 4) #Hex Per Number
	half_point = int((2**quant_size) / 2)
	weights = {}
	progress = tqdm(total=len(config_data['config']['layers'])-1, desc="Dequantizing weights", unit="layer", miniters=1, mininterval=0)
	ptr = 0
	for layer in config_data['config']['layers']:
		layer_data = []
		if layer['class_name'] == 'InputLayer':
			continue
		if layer['class_name'] == 'Dense':
			d1 = layer['build_config']['input_shape'][1]
			d2 = layer['config']['units']
			w = np.array([])
			batches = d2 // pack_size
			if d2 >= pack_size:
				layer_data = bin_data[ptr:ptr+(((((d1+1)*d2)//pack_size)*8) if d2>=pack_size else 0)+(8*(d1+1) if d2%pack_size != 0 else 0)+(((d1+1)*(d2+(d2%2)))*hpn)]
				for i in range(d1):
					w2 = np.array([])
					ptr_2 = (i*batches*(8+batch_shift)) + (i*(d2-(batches*batch_shift)+(d2%2))) + (i*(8 if (batches*batch_shift) < d2 else 0))
					if d2 >= pack_size:
						for j in range(batches):
							scale_hex = layer_data[ptr_2+(j*(8+batch_shift)):(ptr_2+(j*(8+batch_shift)))+8]
							scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
							data_hex = layer_data[(ptr_2+(j*(8+batch_shift)))+8:(ptr_2+(j*(8+batch_shift)))+(8+batch_shift)]
							for k in range(0, batch_shift//2):
								byte = int(data_hex[k*2:(k*2)+2], 16)
								if (quant_size == 4):
									n1 = ((byte >> 4) & 0x0F) - half_point
									w2 = np.append(w2, n1 * scale)
									n2 = (byte & 0x0F)
									if (n2 != 0):
										w2 = np.append(w2, (n2 - half_point) * scale)
								elif (quant_size == 8):
									n = byte - half_point
									w2 = np.append(w2, n * scale)

					if d2 % pack_size != 0:
						irreg_shift = int((d2 % pack_size) * (quant_size / 4))
						scale_hex = layer_data[ptr_2+(batches*(8+batch_shift)):ptr_2+(batches*(8+batch_shift))+8]
						scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
						data_hex = layer_data[ptr_2+(batches*(8+batch_shift))+8:ptr_2+(batches*(8+batch_shift))+(8+irreg_shift)]

						for k in range(0, (irreg_shift // 2) + (irreg_shift % 2)):
							byte = int(data_hex[k*2:(k*2)+2], 16)
							if (quant_size == 4):
								n1 = ((byte >> 4) & 0x0F) - half_point
								w2 = np.append(w2, n1 * scale)
								n2 = (byte & 0x0F)
								if (n2 != 0):
									w2 = np.append(w2, (n2 - half_point) * scale)
							elif (quant_size == 8):
								n = byte - half_point
								w2 = np.append(w2, n * scale)
					if len(w) == 0:
						w = w2
					else:
						w = np.vstack((w, w2))

				ptr_2 = (d1*batches*(8+batch_shift)) + (d1*(d2-(batches*batch_shift)+(d2%2))) + (d1*(8 if (batches*batch_shift) < d2 else 0))
				w3 = np.array([])
				if (d2 >= pack_size):
					for i in range(batches):
						scale_hex = layer_data[ptr_2+(i*(8+batch_shift)):ptr_2+(i*(8+batch_shift))+8]
						scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
						data_hex = layer_data[ptr_2+(i*(8+batch_shift))+8:ptr_2+(i*(8+batch_shift))+(8+batch_shift)]
						for k in range(0, batch_shift//2):
							byte = int(data_hex[k*2:(k*2)+2], 16)
							if (quant_size == 4):
								n1 = ((byte >> 4) & 0x0F) - half_point
								w3 = np.append(w3, n1 * scale)
								n2 = (byte & 0x0F)
								if (n2 != 0):
									w3 = np.append(w3, (n2 - half_point) * scale)
							elif (quant_size == 8):
								n = byte - half_point
								w3 = np.append(w3, n * scale)

				if d2 % pack_size != 0:
					irreg_shift = int((d2 % pack_size) * (quant_size / 4))
					scale_hex = layer_data[ptr_2+(batches*(8+batch_shift)):ptr_2+(batches*(8+batch_shift))+8]
					scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
					data_hex = layer_data[ptr_2+(batches*(8+batch_shift))+8:ptr_2+(batches*(8+batch_shift))+(8+irreg_shift)]
					for k in range(0, (irreg_shift // 2) + (irreg_shift % 2)):
						byte = int(data_hex[k*2:(k*2)+2], 16)
						if (quant_size == 4):
							n1 = ((byte >> 4) & 0x0F) - half_point
							w3 = np.append(w3, n1 * scale)
							n2 = (byte & 0x0F)
							if (n2 != 0):
								w3 = np.append(w3, (n2 - half_point) * scale)
						elif (quant_size == 8):
							n = byte - half_point
							w3 = np.append(w3, n * scale)

				weights[layer['config']['name']] = [w, w3]
			else:
				layer_data = bin_data[ptr:ptr+((d1+1)*d2*8)]
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
			w = np.array([])
			batches = d3 // pack_size
			if d3 >= pack_size:
				layer_data = bin_data[ptr:ptr+((((((d1*d2)+1)*d3)//pack_size)*8) if d3>=pack_size else 0)+(8*((d1*d2)+1) if d3%pack_size != 0 else 0)+((((d1*d2)+1)*(d3+(d3%2)))*hpn)]
				for i in range(d1):
					w2 = np.array([])
					for j in range(d2):
						w3 = np.array([])
						ptr_2 = (i*batches*(8+batch_shift)) + (i*(d3-(batches*batch_shift)+(d3%2))) + (i*(8 if (batches*batch_shift) < d3 else 0))
						if d3 >= pack_size:
							for k in range(batches):
								scale_hex = layer_data[ptr_2+(k*(8+batch_shift)):ptr_2+(k*(8+batch_shift))+8]
								scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
								data_hex = layer_data[(ptr_2+(k*(8+batch_shift)))+8:(ptr_2+(k*(8+batch_shift)))+(8+batch_shift)]
								for l in range(0, batch_shift//2):
									byte = int(data_hex[l*2:(l*2)+2], 16)
									if (quant_size == 4):
										n1 = ((byte >> 4) & 0x0F) - half_point
										w3 = np.append(w3, n1 * scale)
										n2 = (byte & 0x0F)
										if (n2 != 0):
											w3 = np.append(w3, (n2 - half_point) * scale)
									elif (quant_size == 8):
										n = byte - half_point
										w3 = np.append(w3, n * scale)
						if d3 % pack_size != 0:
							irreg_shift = int((d3 % pack_size) * (quant_size / 4))
							scale_hex = layer_data[ptr_2+(batches*(8+batch_shift)):ptr_2+(batches*(8+batch_shift))+8]
							scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
							data_hex = layer_data[ptr_2+(batches*(8+batch_shift))+8:ptr_2+(batches*(8+batch_shift))+(8+irreg_shift)]
							
							for l in range(0, (irreg_shift // 2) + (irreg_shift % 2)):
								byte = int(data_hex[l*2:(l*2)+2], 16)
								if (quant_size == 4):
									n1 = ((byte >> 4) & 0x0F) - half_point
									w3 = np.append(w3, n1 * scale)
									n2 = (byte & 0x0F)
									if (n2 != 0):
										w3 = np.append(w3, (n2 - half_point) * scale)
								elif (quant_size == 8):
									n = byte - half_point
									w3 = np.append(w3, n * scale)
						if len(w2) == 0:
							w2 = w3
						else:
							w2 = np.vstack((w2, w3))
					if len(w) == 0:
						w = w2
					else:
						w = np.vstack((w, w2))

				ptr_2 = ((d1*d2)*batches*(8+batch_shift)) + ((d1*d2)*(d3-(batches*batch_shift)+(d3%2))) + ((d1*d2)*(8 if (batches*batch_shift) < d3 else 0))
				w4 = np.array([])
				if (d3 >= pack_size):
					for i in range(batches):
						scale_hex = layer_data[ptr_2+(i*(8+batch_shift)):ptr_2+(i*(8+batch_shift))+8]
						scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
						data_hex = layer_data[ptr_2+(i*(8+batch_shift))+8:ptr_2+(i*(8+batch_shift))+(8+batch_shift)]
						for k in range(0, batch_shift//2):
							byte = int(data_hex[k*2:(k*2)+2], 16)
							if (quant_size == 4):
								n1 = ((byte >> 4) & 0x0F) - half_point
								w4 = np.append(w4, n1 * scale)
								n2 = (byte & 0x0F)
								if (n2 != 0):
									w4 = np.append(w4, (n2 - half_point) * scale)
							elif (quant_size == 8):
								n = byte - half_point
								w4 = np.append(w4, n * scale)
				
				if d3 % pack_size != 0:
					irreg_shift = int((d3 % pack_size) * (quant_size / 4))
					scale_hex = layer_data[ptr_2+(batches*(8+batch_shift)):ptr_2+(batches*(8+batch_shift))+8]
					scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
					data_hex = layer_data[ptr_2+(batches*(8+batch_shift))+8:ptr_2+(batches*(8+batch_shift))+(8+irreg_shift)]
					for k in range(0, (irreg_shift // 2) + (irreg_shift % 2)):
						byte = int(data_hex[k*2:(k*2)+2], 16)
						if (quant_size == 4):
							n1 = ((byte >> 4) & 0x0F) - half_point
							w4 = np.append(w4, n1 * scale)
							n2 = (byte & 0x0F)
							if (n2 != 0):
								w4 = np.append(w4, (n2 - half_point) * scale)
						elif (quant_size == 8):
							n = byte - half_point
							w4 = np.append(w4, n * scale)
				weights[layer['config']['name']] = [w, w4]
			else:
				layer_data = bin_data[ptr:ptr+(((d1*d2)+1)*d3*8)]
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
			w = np.array([])
			batches = d4 // pack_size
			if d4 >= pack_size:
				layer_data = bin_data[ptr:ptr+((((((d1*d2*d3)+1)*d4)//pack_size)*8) if d4>=pack_size else 0)+(8*((d1*d2*d3)+1) if d4%pack_size != 0 else 0)+((((d1*d2*d3)+1)*(d4+(d4%2)))*hpn)]
				for i in range(d1):
					w2 = np.array([])
					for j in range(d2):
						w3 = np.array([])
						for k in range(d3):
							w4 = np.array([])
							ptr_2 = (i*batches*(8+batch_shift)) + (i*(d4-(batches*batch_shift)+(d4%2))) + (i*(8 if (batches*batch_shift) < d4 else 0))
							if d4 >= pack_size:
								for l in range(batches):
									scale_hex = layer_data[ptr_2+(l*(8+batch_shift)):ptr_2+(l*(8+batch_shift))+8]
									scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
									data_hex = layer_data[(ptr_2+(l*(8+batch_shift)))+8:(ptr_2+(l*(8+batch_shift)))+(8+batch_shift)]
									for m in range(0, batch_shift//2):
										byte = int(data_hex[m*2:(m*2)+2], 16)
										if (quant_size == 4):
											n1 = ((byte >> 4) & 0x0F) - half_point
											w4 = np.append(w4, n1 * scale)
											n2 = (byte & 0x0F)
											if (n2 != 0):
												w4 = np.append(w4, (n2 - half_point) * scale)
										elif (quant_size == 8):
											n = byte - half_point
											w4 = np.append(w4, n * scale)
							if d4 % pack_size != 0:
								irreg_shift = int((d4 % pack_size) * (quant_size / 4))
								scale_hex = layer_data[ptr_2+(batches*(8+batch_shift)):ptr_2+(batches*(8+batch_shift))+8]
								scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
								data_hex = layer_data[ptr_2+(batches*(8+batch_shift))+8:ptr_2+(batches*(8+batch_shift))+(8+irreg_shift)]
								
								for m in range(0, (irreg_shift // 2) + (irreg_shift % 2)):
									byte = int(data_hex[m*2:(m*2)+2], 16)
									if (quant_size == 4):
										n1 = ((byte >> 4) & 0x0F) - half_point
										w4 = np.append(w4, n1 * scale)
										n2 = (byte & 0x0F)
										if (n2 != 0):
											w4 = np.append(w4, (n2 - half_point) * scale)
									elif (quant_size == 8):
										n = byte - half_point
										w4 = np.append(w4, n * scale)
							if len(w3) == 0:
								w3 = w4
							else:
								w3 = np.vstack((w3, w4))
						if len(w2) == 0:
							w2 = w3
						else:
							w2 = np.vstack((w2, w3))
					if len(w) == 0:
						w = w2
					else:
						w = np.vstack((w, w2))

				ptr_2 = ((d1*d2*d3)*batches*(8+batch_shift)) + ((d1*d2*d3)*(d4-(batches*batch_shift)+(d4%2))) + ((d1*d2*d3)*(8 if (batches*batch_shift) < d4 else 0))
				w5 = np.array([])
				if (d4 >= pack_size):
					for i in range(batches):
						scale_hex = layer_data[ptr_2+(i*(8+batch_shift)):ptr_2+(i*(8+batch_shift))+8]
						scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
						data_hex = layer_data[ptr_2+(i*(8+batch_shift))+8:ptr_2+(i*(8+batch_shift))+(8+batch_shift)]
						for k in range(0, batch_shift//2):
							byte = int(data_hex[k*2:(k*2)+2], 16)
							if (quant_size == 4):
								n1 = ((byte >> 4) & 0x0F) - half_point
								w5 = np.append(w5, n1 * scale)
								n2 = (byte & 0x0F)
								if (n2 != 0):
									w5 = np.append(w5, (n2 - half_point) * scale)
							elif (quant_size == 8):
								n = byte - half_point
								w5 = np.append(w5, n * scale)

				if d4 % pack_size != 0:
					irreg_shift = int((d4 % pack_size) * (quant_size / 4))
					scale_hex = layer_data[ptr_2+(batches*(8+batch_shift)):ptr_2+(batches*(8+batch_shift))+8]
					scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
					data_hex = layer_data[ptr_2+(batches*(8+batch_shift))+8:ptr_2+(batches*(8+batch_shift))+(8+irreg_shift)]
					for k in range(0, (irreg_shift // 2) + (irreg_shift % 2)):
						byte = int(data_hex[k*2:(k*2)+2], 16)
						if (quant_size == 4):
							n1 = ((byte >> 4) & 0x0F) - half_point
							w5 = np.append(w5, n1 * scale)
							n2 = (byte & 0x0F)
							if (n2 != 0):
								w5 = np.append(w5, (n2 - half_point) * scale)
						elif (quant_size == 8):
							n = byte - half_point
							w5 = np.append(w5, n * scale)
				weights[layer['config']['name']] = [w, w5]
			else:
				layer_data = bin_data[ptr:ptr+(((d1*d2*d3)+1)*d4*8)]
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
			layer_data = bin_data[ptr:ptr+(((d1//pack_size)*8)+(8 if d1%pack_size != 0 else 0)+((d1+(d1%2))*hpn))*2]
			w = np.array([])
			batches = d1 // pack_size
			ptr_2 = 0
			if (d1 >= pack_size):
				if (d1 >= pack_size):
					for i in range(d1//pack_size):
						scale_hex = layer_data[i*(8+batch_shift):(i*(8+batch_shift))+8]
						scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
						data_hex = layer_data[(i*(8+batch_shift))+8:(i*(8+batch_shift))+(8+batch_shift)]
						for k in range(0, batch_shift//2):
							byte = int(data_hex[k*2:(k*2)+2], 16)
							if (quant_size == 4):
								n1 = ((byte >> 4) & 0x0F) - half_point
								w = np.append(w, n1 * scale)
								n2 = (byte & 0x0F)
								if (n2 != 0):
									w = np.append(w, (n2 - half_point) * scale)
							elif (quant_size == 8):
								n = byte - half_point
								w = np.append(w, n * scale)
				if d1 % pack_size != 0:
					irreg_shift = int((d1 % pack_size) * (quant_size / 4))
					scale_hex = layer_data[batches*(8+batch_shift):(batches*(8+batch_shift))+8]
					scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
					data_hex = layer_data[(batches*(8+batch_shift))+8:(batches*(8+batch_shift))+(8+irreg_shift)]
					for k in range(0, (irreg_shift // 2) + (irreg_shift % 2)):
						byte = int(data_hex[k*2:(k*2)+2], 16)
						if (quant_size == 4):
							n1 = ((byte >> 4) & 0x0F) - half_point
							w = np.append(w, n1 * scale)
							n2 = (byte & 0x0F)
							if (n2 != 0):
								w = np.append(w, (n2 - half_point) * scale)
						elif (quant_size == 8):
							n = byte - half_point
							w = np.append(w, n * scale)
				ptr_2 = batches*(8+batch_shift) + (d1-(batches*batch_shift)+(d1%2)) + (8 if batches*batch_shift < d1 else 0)
				w2 = np.array([])
				if (d1 >= pack_size):
					for i in range(d1//pack_size):
						scale_hex = layer_data[ptr_2+(i*(8+batch_shift)):ptr_2+(i*(8+batch_shift))+8]
						scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
						data_hex = layer_data[ptr_2+(i*(8+batch_shift))+8:ptr_2+(i*(8+batch_shift))+(8+batch_shift)]
						for k in range(0, batch_shift//2):
							byte = int(data_hex[k*2:(k*2)+2], 16)
							if (quant_size == 4):
								n1 = ((byte >> 4) & 0x0F) - half_point
								w2 = np.append(w2, n1 * scale)
								n2 = (byte & 0x0F)
								if (n2 != 0):
									w2 = np.append(w2, (n2 - half_point) * scale)
							elif (quant_size == 8):
								n = byte - half_point
								w2 = np.append(w2, n * scale)
					ptr_2 += batches*(8+batch_shift)
				if d1 % pack_size != 0:
					irreg_shift = int((d1 % pack_size) * (quant_size / 4))
					scale_hex = layer_data[ptr_2+(batches*(8+batch_shift)):ptr_2+(batches*(8+batch_shift))+8]
					scale = struct.unpack('>f', bytes.fromhex(scale_hex))[0]
					data_hex = layer_data[ptr_2+(batches*(8+batch_shift))+8:ptr_2+(batches*(8+batch_shift))+(8+irreg_shift)]
					for k in range(0, (irreg_shift // 2) + (irreg_shift % 2)):
						byte = int(data_hex[k*2:(k*2)+2], 16)
						if (quant_size == 4):
							n1 = ((byte >> 4) & 0x0F) - half_point
							w2 = np.append(w2, n1 * scale)
							n2 = (byte & 0x0F)
							if (n2 != 0):
								w2 = np.append(w2, (n2 - half_point) * scale)
						elif (quant_size == 8):
							n = byte - half_point
							w2 = np.append(w2, n * scale)
				weights[layer['config']['name']] = [w, w2]
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

		progress.update(1)

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