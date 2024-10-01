# Using the approximate solver

The solver requires the data matrix `X` which has the size `n*p`, with `n` being the number of samples and `p` being the data dimension.

First, you will need to normalize `X`, which can be done as

```
from l0bnb2 import preprocess
_,_,_,_,Y,_ = preprocess(X, assume_centered = False)
```
This will ensure `X` is centered and is normalized by `sqrt(n)`.  The approximate solver then can be called as
```
from l0bnb2 import heuristic_solve
Theta_approx, _, _, _  =  heuristic_solve(Y, l0, l2, M)
```
Here, `l0, l2` are L0 and L2 regularization coefficients, respectively. `M` is the Big-M parameter.

The method `heuristic_solve` allows for further customizations of the optimization. For example, you can use `cd_max_itr` to set the maximum number of coordinate descent iterations, and `rel_tol` to set the relative error tolerance for convergence. You can use the argument `solver` to select the solver algorithm which will default to coordinate descent with partial swap local search.

# Using the BnB solver

You can also run the BnB exact solver as follows, given the data matrix `X`:

```
from  l0bnb2  import  BNBTree
tree  =  BNBTree(X_train)
sol  =  tree.solve(l0, l2, M, warm_start=Theta_approx, verbose=True, gap_tol=5e-2, time_limit=120)
Theta_rec  =  sol.Theta
```

You can pass in a warm-start to the BnB solver via `warm_start`, which here, is set to the solution from the approximate solver from before. Setting the option `verbose` to true will print out the BnB progress. You can set the MIP gap tolerance by `gap_tol` and the BnB time limit (in seconds) via `time_limit`. 

The method `tree.solve` has further options for customization of the optimization. For example, you can perform Depth-first search in BnB by setting `number_of_dfs_levels`  (the default is zero, BFS). You can also set the upper bound (approximate) solver via `upper_solver` (the default is coordinate descent with partial swap local search). 

# Data generation

Synthetic data generation can be done via `generate_synthetic`.  Please refer to the following examples.

## Example 1

```
from  l0bnb2  import generate_synthetic
s  =  3
p  =  20  
n  =  200  
model  =  "banded_Toeplitz_precision"  
normalize  =  True
X_train, Sigma_truth, Theta_truth  =  generate_synthetic(n, p, model, normalize, rng=0, cond=2, half_bandwidth=s)
```
This example generates `n=200` samples from a `p=20`-dimensional Normal distribution. The precision matrix of this distribution follows a banded Toeplitz model.  The half-bandwidth of this precision matrix is given by `s=3`, and the condition number of this matrix is given by `cond = 2`.  By setting `normalize=True`, the data is normalized to have maximum variance of 1. 

## Example 2

```
from  l0bnb2  import generate_synthetic
s  =  3
p  =  20  
n  =  200  
model  =  "uniform"  
normalize  =  True
X_train, Sigma_truth, Theta_truth  =  generate_synthetic(n, p, model, normalize, rng=0, cond=2, p0=s/p)
```
This example is similar to the previous one, except that each off-diagonal coordinate of the precision matrix can be nonzero with probability `p0=s/p` independently.

# Demo
Please refer to `demo.py` for an example of running the pipeline.