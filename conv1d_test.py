import kagglehub
path = kagglehub.dataset_download("antfilatov/mini-speech-commands")

import os
data_root = os.path.join(path, "mini_speech_commands")
	
commands = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
num_classes = len(commands)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from uniquant import quantize, dequantize, dequantize_save
from tqdm.auto import tqdm
import json
import time

train_wav_ds = tf.keras.utils.audio_dataset_from_directory(
	directory=data_root,
	batch_size=64,
	validation_split=0.2,
	subset="training",
	seed=86,
	label_mode="int",
	class_names=commands,
	output_sequence_length=16000
)

val_wav_ds = tf.keras.utils.audio_dataset_from_directory(
	directory=data_root,
	batch_size=64,
	validation_split=0.2,
	subset="validation",
	seed=86,
	label_mode="int",
	class_names=commands,
	output_sequence_length=16000,
)

val_batches = tf.data.experimental.cardinality(val_wav_ds).numpy()
test_wav_ds = val_wav_ds.take(val_batches // 2)
val_wav_ds = val_wav_ds.skip(val_batches // 2)

AUTOTUNE = tf.data.AUTOTUNE

def waveform_to_spec(waveform, label):
	waveform = tf.squeeze(waveform, axis=-1) 

	stft = tf.signal.stft(
		waveform,
		frame_length=256,
		frame_step=128,
		fft_length=256,
		window_fn=tf.signal.hann_window,
		pad_end=True,
	)
	spec = tf.abs(stft)

	spec = tf.math.log(spec + 1e-6)

	mean = tf.reduce_mean(spec, axis=[1, 2], keepdims=True)
	std = tf.math.reduce_std(spec, axis=[1, 2], keepdims=True) + 1e-6
	spec = (spec - mean) / std

	return spec, label

train_ds = train_wav_ds.map(waveform_to_spec, num_parallel_calls=AUTOTUNE).cache().shuffle(10000).prefetch(AUTOTUNE)
val_ds = val_wav_ds.map(waveform_to_spec, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
test_ds = test_wav_ds.map(waveform_to_spec, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

for xb, yb in train_ds.take(1):
	input_shape = xb.shape[1:]

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

model.add(tf.keras.layers.Conv1D(512, 7, padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('gelu'))
model.add(tf.keras.layers.MaxPooling1D(2))

model.add(tf.keras.layers.Conv1D(1024, 5, padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('gelu'))
model.add(tf.keras.layers.MaxPooling1D(2))

model.add(tf.keras.layers.Conv1D(2048, 3, padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('gelu'))
model.add(tf.keras.layers.MaxPooling1D(2))

model.add(tf.keras.layers.Conv1D(4096, 3, padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('gelu'))

model.add(tf.keras.layers.GlobalAveragePooling1D())

model.add(tf.keras.layers.Dense(4096, activation='gelu'))
model.add(tf.keras.layers.LayerNormalization())
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(2048, activation='gelu'))
model.add(tf.keras.layers.LayerNormalization())
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(1024, activation='gelu'))
model.add(tf.keras.layers.LayerNormalization())

model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_ds, epochs=10, validation_data=val_ds)

def eval_ds(m, ds):
	y_true = []
	y_pred = []
	for xb, yb in ds:
		prob = m.predict(xb, verbose=0)
		pred = np.argmax(prob, axis=1)
		y_true.append(yb.numpy())
		y_pred.append(pred)
	y_true = np.concatenate(y_true, axis=0)
	y_pred = np.concatenate(y_pred, axis=0)

	acc = accuracy_score(y_true, y_pred)
	prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
	rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
	f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
	return acc, prec, rec, f1

results = []

accuracy, precision, recall, f1 = eval_ds(model, test_ds)
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
results.append({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})

model.save("model.keras", overwrite=True)
del model

quant_t = [4, 8]
num_t = [8, 16, 32, 64, 128]
progress = tqdm(total=len(quant_t)*len(num_t), desc="Tests", unit="test", miniters=1, mininterval=0)
for quant_size in quant_t:
	for num in num_t:
		start = time.perf_counter()
		quantize("model.keras", quant_name='m_c1d'+str(num)+"_"+str(quant_size), overwrite=True, pack_size=num, quant_size = quant_size)
		qtime = time.perf_counter() - start
		start = time.perf_counter()
		model = dequantize('m_c1d'+str(num)+"_"+str(quant_size)+".uniq")
		dqtime = time.perf_counter() - start
		accuracy, precision, recall, f1 = eval_ds(model, test_ds)
		print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
		results.append({"quant_size": quant_size, "pack_size": num, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "quant_time": qtime, "dequant_time": dqtime})
		del model
		progress.update(1)
	
with open(f"test_conv1d.json", "w") as f:
	json.dump(results, f, indent=4)