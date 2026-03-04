import kagglehub
path = kagglehub.dataset_download("ai4a-lab/herb-plant-classification-dataset")

results = []

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from uniquant import quantize, dequantize
from tqdm.auto import tqdm

def _is_img(p): return p.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp"))
root = path
if len([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]) == 1:
	only = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))][0]
	if len([d for d in os.listdir(os.path.join(root, only)) if os.path.isdir(os.path.join(root, only, d))]) > 1: root = os.path.join(root, only)

class_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
class_to_idx = {c:i for i,c in enumerate(class_names)}
file_paths, labels = [], []
for c in class_names:
	cd = os.path.join(root, c)
	for fn in os.listdir(cd):
		fp = os.path.join(cd, fn)
		if os.path.isfile(fp) and _is_img(fp): file_paths.append(fp); labels.append(class_to_idx[c])
file_paths, labels = np.array(file_paths), np.array(labels, dtype=np.int64)

X_train, X_test, y_train, y_test = train_test_split(file_paths, labels, test_size=0.2, random_state=86, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=86, stratify=y_train)

IMG_H, IMG_W = 256, 256
def _load_img(p, y):
	b = tf.io.read_file(p); x = tf.image.decode_image(b, channels=3, expand_animations=False)
	x = tf.image.resize(x, (IMG_H, IMG_W), method="bilinear"); x = tf.cast(x, tf.float32) / 255.0; y = tf.cast(y, tf.int32)
	x.set_shape((IMG_H, IMG_W, 3))
	return x, y

def _aug_img(x, y):
	x = tf.image.random_flip_left_right(x)
	x = tf.image.random_brightness(x, 0.10)
	x = tf.image.random_contrast(x, 0.85, 1.15)
	x = tf.image.random_saturation(x, 0.85, 1.15)
	x = tf.clip_by_value(x, 0.0, 1.0)
	return x, y

NUM_CLASSES = len(class_names)
def _to_onehot(x, y): return x, tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)
def _to_onehot_sm(x, y):
	y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES); eps = tf.constant(0.10, tf.float32)
	return x, y*(1.0-eps)+eps/tf.cast(NUM_CLASSES, tf.float32)


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(min(len(X_train), 4096), seed=86, reshuffle_each_iteration=True).map(_load_img, num_parallel_calls=tf.data.AUTOTUNE).map(_aug_img, num_parallel_calls=tf.data.AUTOTUNE).map(_to_onehot_sm, num_parallel_calls=tf.data.AUTOTUNE).batch(16).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).map(_load_img, num_parallel_calls=tf.data.AUTOTUNE).map(_to_onehot, num_parallel_calls=tf.data.AUTOTUNE).batch(16).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).map(_load_img, num_parallel_calls=tf.data.AUTOTUNE).map(_to_onehot, num_parallel_calls=tf.data.AUTOTUNE).batch(16).prefetch(tf.data.AUTOTUNE)

model = keras.Sequential()
model.add(keras.layers.InputLayer(shape=(IMG_H, IMG_W, 3)))
model.add(keras.layers.Conv2D(64, 3, padding='same')); model.add(keras.layers.BatchNormalization()); model.add(keras.layers.Activation('gelu'))
model.add(keras.layers.Conv2D(64, 3, padding='same')); model.add(keras.layers.BatchNormalization()); model.add(keras.layers.Activation('gelu'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(128, 3, padding='same')); model.add(keras.layers.BatchNormalization()); model.add(keras.layers.Activation('gelu'))
model.add(keras.layers.Conv2D(128, 3, padding='same')); model.add(keras.layers.BatchNormalization()); model.add(keras.layers.Activation('gelu'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(256, 3, padding='same')); model.add(keras.layers.BatchNormalization()); model.add(keras.layers.Activation('gelu'))
model.add(keras.layers.Conv2D(256, 3, padding='same')); model.add(keras.layers.BatchNormalization()); model.add(keras.layers.Activation('gelu'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(512, 3, padding='same')); model.add(keras.layers.BatchNormalization()); model.add(keras.layers.Activation('gelu'))
model.add(keras.layers.Conv2D(512, 3, padding='same')); model.add(keras.layers.BatchNormalization()); model.add(keras.layers.Activation('gelu'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(8192, activation='gelu')); model.add(keras.layers.Dropout(0.40))
model.add(keras.layers.Dense(4096, activation='gelu')); model.add(keras.layers.Dropout(0.30))
model.add(keras.layers.Dense(2048, activation='gelu')); model.add(keras.layers.Dropout(0.20))
model.add(keras.layers.Dense(1024, activation='gelu'))
model.add(keras.layers.Dense(len(class_names), activation='softmax'))

model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=2e-4, clipnorm=1.0), loss=keras.losses.CategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')])
model.fit(train_ds, epochs=45, validation_data=val_ds)

y_pred_prob = model.predict(test_ds)
y_pred = np.argmax(y_pred_prob, axis=1)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
results.append({"accuracy": float(accuracy), "precision": float(precision), "recall": float(recall), "f1": float(f1)})

model.save("model.keras", overwrite=True)
del model
del X_train
del y_train


#results.append({"accuracy": 0.7321867321867321, "precision": 0.7562410904327956, "recall": 0.7101213223410621, "f1": 0.7139587267877459})

quant_t = [4, 8]
num_t = [8, 16, 32, 64, 128]
progress = tqdm(total=len(quant_t) * len(num_t), desc="Tests", unit="test", miniters=1, mininterval=0)
for quant_size in quant_t:
	for num in num_t:
		quantize("model.keras", quant_name='m_c2d'+str(num)+"_"+str(quant_size), overwrite=True, pack_size=num, quant_size = quant_size)
		model = dequantize("m_c2d"+str(num)+"_"+str(quant_size) + ".uniq")
		y_pred_prob = model.predict(test_ds)
		y_pred = np.argmax(y_pred_prob, axis=1)
		accuracy = accuracy_score(y_test, y_pred)
		precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
		recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
		f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
		print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
		results.append({"quant_size": quant_size, "pack_size": num, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
		del model
		progress.update(1)

with open("test_conv2d.json", "w") as f:
	json.dump(results, f, indent=4)