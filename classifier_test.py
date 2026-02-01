import kagglehub
path = kagglehub.dataset_download("mohankrishnathalla/medical-insurance-cost-prediction")

import pandas as pd
data = pd.read_csv(path + "/medical_insurance.csv")
data.loc[1]
data.dropna()

results = []

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from uniquant import quantize, dequantize, dequantize_save
from tqdm.auto import tqdm
import json

X = data.drop('is_high_risk', axis=1)
y = data['is_high_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=86)

categorical_features = ['sex', 'region', 'urban_rural', 'education', 'marital_status', 'employment_status', 'smoker', 'alcohol_freq', 'plan_type', 'network_tier']
numerical_features = [col for col in X.columns if col not in categorical_features]
preprocessor = ColumnTransformer(
	transformers=[
		('num', StandardScaler(), numerical_features),
		('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
	])
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
input_dim = X_train.shape[1]

model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(input_dim,)))
model.add(keras.layers.Dense(8192, activation='gelu'))
model.add(keras.layers.LayerNormalization())
model.add(keras.layers.Dense(4096, activation='gelu'))
model.add(keras.layers.LayerNormalization())
model.add(keras.layers.Dense(4096, activation='gelu'))
model.add(keras.layers.LayerNormalization())
model.add(keras.layers.Dense(2048, activation='gelu'))
model.add(keras.layers.LayerNormalization())
model.add(keras.layers.Dense(1024, activation='gelu'))
model.add(keras.layers.LayerNormalization())
model.add(keras.layers.Dense(512, activation='gelu'))
model.add(keras.layers.LayerNormalization())
model.add(keras.layers.Dense(256, activation='gelu'))
model.add(keras.layers.LayerNormalization())
model.add(keras.layers.Dense(128, activation='gelu'))
model.add(keras.layers.LayerNormalization())
model.add(keras.layers.Dense(64, activation='gelu'))
model.add(keras.layers.LayerNormalization())
model.add(keras.layers.Dense(32, activation='gelu'))
model.add(keras.layers.LayerNormalization())
model.add(keras.layers.Dense(16, activation='gelu'))
model.add(keras.layers.LayerNormalization())
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adamw', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
results.append({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
model.save("model.keras", overwrite=True)
del model
del X_train
del y_train

progress = tqdm(total=10, desc="Tests", unit="test", miniters=1, mininterval=0)
for quant_size in [4, 8]:
	for num in [32,64,16,8,128]:
		quantize("model.keras", quant_name='m_cls'+str(num)+"_"+str(quant_size), overwrite=True, pack_size=num, quant_size = quant_size)
		model = dequantize('m_cls'+str(num)+"_"+str(quant_size)+".uniq")
		y_pred_prob = model.predict(X_test).flatten()
		y_pred = (y_pred_prob > 0.5).astype(int)
		accuracy = accuracy_score(y_test, y_pred)
		precision = precision_score(y_test, y_pred)
		recall = recall_score(y_test, y_pred)
		print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
		results.append({"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
		del model
		progress.update(1)
	
	with open(f"test_classifier_{quant_size}bit_results_32_64_16_8_128.json", "w") as f:
		json.dump(results, f, indent=4)