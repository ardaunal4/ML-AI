import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv("D:\PYTHON\ML-AI\polynomial-linear-regression\polynomial_regression.csv", sep = ";")

x = df.araba_fiyat.values.shape(-1, 1)
y = df.araba_max_hiz.values.shape(-1, 1)

polynomial_regression = PolynomialFeatures(degree = 5)
x_poly_reg = polynomial_regression.fit_transform(x)

linear_regression = LinearRegression()
linear_regression.fit(x_poly_reg)
y_head = linear_regression.predict(x_poly_reg)

plt.scatter(x, y, color = "blue")
plt.plot(x, y_head, color = "red")
plt.show()
