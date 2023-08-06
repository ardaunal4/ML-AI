import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("D:\PYTHON\ML-AI\multiple-linear-regression\multiple_linear_regression_dataset.csv", sep = ";")

multiple_linear_regression = LinearRegression()

x = df.iloc[:, [0, 2]].values
y = df.maas.values.reshape(-1, 1)

multiple_linear_regression.fit(x, y)

# Predictions:
b0 = multiple_linear_regression.intercept_
print("Intersection point: ", b0)
coefs = multiple_linear_regression.coef_
print("Coefficients: ", coefs)
