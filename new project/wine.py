import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib as jb

df = pd.read_csv("winequality-red.csv", sep=';')
print(df)

x = df[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
y = df["quality"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = Sequential()
model.add(Dense(10, activation="relu", input_shape=(11,)))
model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=50)

jb.dump(model, 'redwine_model.pkl')
print("Training Successful")

result=model.predict(x_test)
print("MSE",mean_squared_error(y_test,result))
print("RMSE",math.sqrt(mean_squared_error(y_test,result)))
print("R2 Score",r2_score(y_test,result))