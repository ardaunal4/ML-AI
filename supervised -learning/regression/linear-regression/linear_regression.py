import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("D:\PYTHON\ML-AI\linear-regression\linear_regression_dataset.csv", sep = ";")

linear_regression = LinearRegression()

x = df.deneyim.values.reshape(-1, 1)
y = df.maas.values.reshape(-1, 1)

linear_regression.fit(x, y)

# Predictions:
b0 = linear_regression.intercept_
print("Intersection point: ", b0)
b1 = linear_regression.coef_
print("Coefficient(slope): ", b1)

y_head = linear_regression.predict(x)
# R^2
print("R^2 score = ", r2_score(y, y_head))

# Visualization
x_predict = np.arange(0, max(x) + 1, 1).reshape(-1, 1)
y_predict = linear_regression.predict(x_predict)

plt.figure(figsize = (10, 8))
plt.scatter(x, y, color = "blue")
plt.plot(x_predict, y_predict, color = "red")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()