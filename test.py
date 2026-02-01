import kagglehub
path = kagglehub.dataset_download("mohankrishnathalla/medical-insurance-cost-prediction")

import pandas as pd
data = pd.read_csv(path + "/medical_insurance.csv")
data.loc[1]
data.dropna()

results = []

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
import numpy as np
from uniquant import quantize, dequantize, dequantize_save
from tqdm.auto import tqdm
import json

X = data.drop('risk_score', axis=1)
y = data['risk_score']
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
model.add(keras.layers.Dense(1))

model.compile(optimizer='adamw', loss='mean_squared_error', metrics=['mean_absolute_error'])

model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

y_pred = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
results.append({"mae": mae, "mse": mse, "rmse": rmse})
model.save("model.keras", overwrite=True)
del(model)
del(X_train)
del(y_train)

progress = tqdm(total=10, desc="Tests", unit="test", miniters=1, mininterval=0)
for quant_size in [4]:
	for num in [32,64,16,8,128]:
		quantize("model.keras", quant_name='m'+str(num)+"_"+str(quant_size), overwrite=True, pack_size=num, quant_size=quant_size)
		#model = dequantize("m"+str(num)+"_"+str(quant_size)+".uniq")
		#y_pred = model.predict(X_test).flatten()
		#mae = mean_absolute_error(y_test, y_pred)
		#mse = mean_squared_error(y_test, y_pred)
		#rmse = np.sqrt(mse)
		#print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
		#results.append({"mae": mae, "mse": mse, "rmse": rmse})
		#del model
		#progress.update(1)

	#with open(f"test_results_{quant_size}bit_32_64_16_8_128.json", "w") as f:
		#json.dump(results, f, indent=4)