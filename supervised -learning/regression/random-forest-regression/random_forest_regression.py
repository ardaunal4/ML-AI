import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("D:\\PYTHON\\ML-AI\\random-forest-regression\\random_forest_regression_dataset.csv", sep = ";", header = None)

x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.reshape(-1, 1)

rf_reg = RandomForestRegressor(n_estimators = 100, random_state = 42) # n_estimators represent number of trees in the algorithm
rf_reg.fit(x, np.ravel(y))

print("A precition: ", rf_reg.predict(np.array([7.5]).reshape(-1, 1)))

y_head = rf_reg.predict(x)
# R^2 score of the model
print("R^2 Score = ", r2_score(y, y_head))

x_head = np.arange(min(x), max(x), 0.01).reshape(-1, 1)
y_head = rf_reg.predict(x_head)

plt.figure(figsize = (10, 8))
plt.scatter(x, y, color = "red")
plt.plot(x_head, y_head, color= "green")
plt.show()

