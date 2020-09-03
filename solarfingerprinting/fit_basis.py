"""

"""

import cvxpy as cvx
import numpy as np

def estimate_parameters(y, mat1, mat2, gamma=0.1, beta=0.1):
    th1 = cvx.Variable(mat1.shape[1])
    th2 = cvx.Variable(mat2.shape[1])
    if gamma is None:
        gamma = 2 * np.sqrt(2 * np.log(len(y)))
    cost = (cvx.sum_squares(y - cvx.matmul(mat1, th1) - cvx.matmul(mat2, th2))
            + gamma * cvx.norm1(th2) + beta * cvx.norm(th1[1:]))
    problem = cvx.Problem(cvx.Minimize(cost))
    problem.solve()
    # th1.value[np.abs(th1.value) <= 1e-2 * th1.value[0]] = 0
    # th2.value[np.abs(th2.value) <= 1e-2 * np.max(th2.value)] = 0
    # sparsity_pattern_1 = np.abs(th1.value) <= 1e-2
    sparsity_pattern_2 = np.abs(th2.value) <= 1e-2
    cost = (cvx.sum_squares(y - cvx.matmul(mat1, th1) - cvx.matmul(mat2, th2))
            + beta * cvx.norm(th1[1:]))
    constraints = []
    # if np.sum(sparsity_pattern_1) > 0:
    #     constraints.append(th1[sparsity_pattern_1] == 0)
    if np.sum(sparsity_pattern_2) > 0:
        constraints.append(th2[sparsity_pattern_2] == 0)
    problem = cvx.Problem(cvx.Minimize(cost), constraints)
    problem.solve()
    # th1.value[np.abs(th1.value) < 1e-4] = 0
    th2.value[np.abs(th2.value) < 1e-4] = 0
    return th1.value, th2.value