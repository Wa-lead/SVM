import numpy as np
from scipy.optimize import minimize
from sklearn import datasets
from numpy.linalg import norm
import matplotlib.pyplot as plt
from math import prod

#Initilize dummy data
X, y = datasets.make_blobs(
    n_samples=3, n_features=2, centers=2, cluster_std=1.05, random_state=40
)

#classify the labels to (1,-1)
y = np.where(y == 0, -1, 1)

#Randmon value for intercept
b = 0

#Objective function
def objective(x):
    x0 = x[0]
    x1 = x[1]
    return 0.5 * x.T @ x


#constraint: for all point in the dataset, the prediciton using the current weight must be true:
#must come up with a way to form hte constraints through loop ( check appendix )

def constraint1(x): return y[0] * (x.T @ X[0] - b) - 1
def constraint2(x): return y[1] * (x.T @ X[1] - b) - 1
def constraint3(x): return y[2] * (x.T @ X[2] - b) - 1


cons = [{'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3}]




solution = minimize(objective, np.array([0, 0]), constraints=cons)
print(solution.x)
#----------- Appendix 

# constraints = []
# for i in range(len(X)):
#     constraints.append(
#         {'type': 'ineq', 'fun': lambda x: y[i] * (x.T @ X[i] - b) - 1}
#     )
#####
# cons2 = [{'type': 'ineq', 'fun': lambda x: y[0] * (x.T @ X[0] - b) - 1},
#          {'type': 'ineq', 'fun': lambda x: y[1] * (x.T @ X[1] - b) - 1},
#          {'type': 'ineq', 'fun': lambda x: y[2] * (x.T @ X[2] - b) - 1}]
