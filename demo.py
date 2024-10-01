import numpy as np
import math as math




from l0bnb2 import BNBTree, generate_synthetic, heuristic_solve, preprocess
from test_util import check_support



# Data generation model
s = 3
p = 20  # Data dimension
n = 200  # Number of samples
model = "banded_Toeplitz_precision"  # Banded precision matrix theta_{i,j} = 0.5^{|i-j|}, half bandwidth = s
model = "uniform"  # Random sparsity mask, (i,j) is nonzero with probability s/p.
normalize = True






# Generate synthetic train data
X_train, Sigma_truth, Theta_truth = generate_synthetic(n, 
                                                 p, 
                                                 model,
                                                 normalize, 
                                                 rng=0, 
                                                 p0=s/p, 
                                                 cond=2, 
                                                 half_bandwidth=s,
                                                 )




# Big-M parameter
M =  np.max(np.abs(Theta_truth))
# L0 and L2 regularization parameters
l0 = 0.05
l2 = 0.05


# Data normalization
_,_,_,_,Y_train,_ = preprocess(X_train, assume_centered = False)


# Approximate solver
Theta_approx, _, _, _ = heuristic_solve(Y_train, l0, l2, M)


# BnB solver 
tree = BNBTree(X_train)
sol = tree.solve(l0, l2, M, warm_start=Theta_approx, verbose=True, gap_tol=5e-2, time_limit=120)  # gap_tol: MIP gap tolerance
Theta_rec = sol.Theta

            
# Solution Quality metrics  
Estimation_error = np.linalg.norm(Theta_rec-Theta_truth, 'fro') / np.linalg.norm(Theta_truth, 'fro')

fp, fn, nnz_rec,nnz_true, mcc = check_support(Theta_truth, Theta_rec, p)  # Support recovery statistics

print(f"Estimation Error of Theta_bnb: {Estimation_error}")
print(f"MCC of Theta_bnb: {mcc}")



 
