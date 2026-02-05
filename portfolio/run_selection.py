import numpy as np
import math as math
import sys

sys.path.append('..')

from GraphL0.l0bnb2 import heuristic_solve, preprocess2



def portfolio_gurobi(Y, X_train, X_test, r_min):
    # Solve portfolio selection with minimum return constraint using Gurobi
    n, num_vars = X_train.shape
    import gurobipy as gp
    from gurobipy import GRB
    m = gp.Model("QP_example")
    x = m.addVars(num_vars, lb=0, ub=GRB.INFINITY, name="x")
    m.setObjective(gp.quicksum(Y[i, j] * x[i] * x[j] for i in range(num_vars) for j in range(num_vars)) , GRB.MINIMIZE)
    m.addConstr(gp.quicksum(x[i] for i in range(num_vars)) == 1, "constraint1")
    m.addConstr(gp.quicksum(X_train[i, j] * x[j] for j in range(num_vars) for i in range(n))   >= r_min, "constraint2")
    m.setParam("OutputFlag", 0)
    m.optimize()
    solution = np.array([x[i].x for i in range(num_vars)])
    r = np.matmul(X_test, solution)
    return np.sum(r), np.std(r)


def pareto_front(x, y):
    # Find the pareto front of return and risk
    points = np.column_stack((x, y))
    n = len(points)
    is_efficient = np.ones(n, dtype=bool)

    for i, (xi, yi) in enumerate(points):
        if is_efficient[i]:
            dominates = ((points[:, 0] >= xi) & (points[:, 1] <= yi)) & (
                (points[:, 0] > xi) | (points[:, 1] < yi)
            )
            if np.any(dominates):
                is_efficient[i] = False
    return points[is_efficient]




X0 = np.loadtxt("portfolio/data.csv", delimiter=",")
X_train = X0[::2]
X_test = X0[1::2]
X0 = None


returns = []
risks = []
RETURNS = [0, 30,  40,  50,  60,  70, 80, 90, 100, 110, 120]


    
_,_,Y_train,_, _ = preprocess2(X_train,X_train, X_train, assume_centered = False)

for l0 in [0.0001, 0.001, 0.1, 1]:
    for l2 in [0.0001, 0.001, 0.1, 1]:

        
        Theta_h, _, _, _ = heuristic_solve(Y_train, l0, l2, 2, solver="L0L2_ASCDPSI", support_type="all",  z=None, \
                        S_diag=None, rel_tol=1e-6, cd_max_itr=100, verbose=False, kkt_max_itr=10, cd_tol=1e-4)

        Sigma_h = np.linalg.inv(Theta_h)

        for ret in RETURNS:
            ret, risk = portfolio_gurobi(Sigma_h, X_test, X_test, ret)
            returns.append(ret)
            risks.append(risk)
            
order = np.argsort(returns)
returns = np.asarray(returns)[order]
risks = np.asarray(risks)[order]

points = pareto_front(returns, risks)


import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(points[:,0], points[:,1], color='b', linewidth=2.5, linestyle="-")
ax.tick_params(axis="both", which="major", labelsize=16)
plt.show()