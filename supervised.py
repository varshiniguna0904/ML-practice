import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
print("ML environment ready")
from sklearn.datasets import load_diabetes 
dataset = load_diabetes(as_frame=True)
df = dataset.frame
print(df.shape)
df.head()
X = df.drop(columns=["target"]) 
y = df["target"] 
print("X shape:", X.shape)
print("y shape:", y.shape)
print(X.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LinearRegression 
X1 = X[["bmi"]] 
Xtr, Xte, ytr, yte = train_test_split(X1, y, test_size=0.2, random_state=42)
simple = LinearRegression() 
simple.fit(Xtr, ytr)
pred = simple.predict(Xte)
print("Slope:", simple.coef_[0]) 
print("Intercept:", simple.intercept_)
print("First 5 predictions:", pred[:5])
pred_train = simple.predict(Xtr)
order = np.argsort(Xtr["bmi"].values) 
plt.scatter(Xtr["bmi"], ytr, label="Actual") 
plt.plot(Xtr["bmi"].values[order],pred_train[order],color="red", label="Model line" ) 
plt.xlabel("BMI") 
plt.ylabel("Target")
plt.legend() 
plt.show()
multiple = LinearRegression()
multiple.fit(X_train, y_train)
pred_multi = multiple.predict(X_test)
print("Coefficients:", multiple.coef_)
print("Intercept:", multiple.intercept_)
print("First predictions:", pred_multi[:5])
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
poly = make_pipeline(
PolynomialFeatures(degree=2, include_bias=False),
LinearRegression()
)
poly.fit(X_train[["bmi"]], y_train)
poly_pred = poly.predict(X_test[["bmi"]])
print(poly_pred[:5])
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, pred_multi)
print("MSE:", mse)
