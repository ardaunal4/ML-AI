import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("D:\PYTHON\ML-AI\decision_tree_regression\decision_tree_regression_dataset.csv", sep = ";", header = None)

x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.reshape(-1, 1)

decision_tree_regression = DecisionTreeRegressor()
decision_tree_regression.fit(x, y)
y_head = decision_tree_regression.predict(x)

plt.scatter(x, y, color = "blue")
plt.plot(x, y_head, color = "red")
plt.show()