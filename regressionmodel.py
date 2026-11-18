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
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test, pred_multi)
print("MSE:", mse)
x = np.array([1,2,3,4,5], dtype=float)
y_demo = np.array([2,4,6,8,10], dtype=float)
m, b = 0.0, 0.0
lr = 0.01
for epoch in range(1000):
     yhat = m*x + b
dm = (-2/len(x))*np.sum(x*(y_demo-yhat))
db = (-2/len(x))*np.sum(y_demo-yhat)
m -= lr*dm
b -= lr*db
print(round(m, 2), round(b, 2))
print("Prediction at x=6:", round(m*6+b, 2))
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
models = {
"Ridge": Ridge(alpha=1.0),
"Lasso": Lasso(alpha=0.1, max_iter=10000),
"ElasticNet": ElasticNet(alpha=0.1, max_iter=10000)
}
for name, alg in models.items():
    pipe = make_pipeline(StandardScaler(), alg)
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    print(name, "R2:", round(r2_score(y_test,p),3))
import statsmodels.api as sm
X_sm = sm.add_constant(X_train)
ols = sm.OLS(y_train, X_sm).fit()
print(ols.summary())
residuals = y_test - pred_multi
plt.scatter(pred_multi, residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
for degree in [1, 2, 5, 10]:
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    model.fit(X_train[["bmi"]], y_train)
    tr = model.predict(X_train[["bmi"]])
    te = model.predict(X_test[["bmi"]])
    print(degree,
          mean_squared_error(y_train,tr),
          mean_squared_error(y_test,te))
from sklearn.metrics import(
mean_absolute_error,
mean_squared_error,
r2_score
)
mae = mean_absolute_error(y_test, pred_multi)
mse = mean_squared_error(y_test, pred_multi)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred_multi)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)
