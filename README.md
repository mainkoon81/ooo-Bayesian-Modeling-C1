# ooo-Minkun-Model-Collection-V-
Non-parametric Bayesian Model

## Background Study
### > Gaussian Story
<img src="https://user-images.githubusercontent.com/31917400/73613995-07e46700-45f3-11ea-8760-6ae349c15dd8.png" />


### > Dirichlet Story
It's a distribution on probability distributions. It's a distribution over `n` dimensional vectors called "θ". It can be thought of as a multivariate beta distribution for a collection of probabilities (that must sum to 1). 
 - Dirichlet distribution is the conjugate prior for the **multinomial likelihood**.
 - Each `θ_i` has its own `α`...weight for each distribution of `θ_i`
 - Each `θ_i` has its own distribution...so each is a function. 
 - Total sum of `θ_i` is 1.
<img src="https://user-images.githubusercontent.com/31917400/73609223-77daf900-45c3-11ea-97b6-52158fec1ba0.png" />

--------------------------------------------------------------------------------------------------------------------

## 1. Gaussian Process and Non-linear Problem
For any set `S`, **GP on `S`** refers to a bunch of random variables(pdf functions: `f(x)`) whose index is the member of the set `S` such that they can have the following properties: The bunch of variables(pdf functions: `f(x)`) are normally multivariate distributed! 

It is a distribution over functions. 
<img src="https://user-images.githubusercontent.com/31917400/73618695-60325d80-4621-11ea-8584-e57f3d37c1de.png" />

To make the prediction, take the point (x, f(x)), then try to generate the mean and CI(or cov): Given that having training set, I try to combine the data with the prior of the functions(with mean and cov)..to make my functions smooth. Use a similarity matrix such that our function approximator makes the two points close by when we fit new data to make sure the two hights are also close by.    
<img src="https://user-images.githubusercontent.com/31917400/73652179-2d6a8280-467e-11ea-8651-2c115fe2c1e7.png" />


```
from __future__ import division # from a module "__future__"
import numpy as np
import matplotlib.pyplot as plt

# GaussianProcess' squared exponential
def kernel(a, b) 
   SQdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2*np.dot(a, b.T)
   return(np.exp(-0.5 * SQdist)

n = 50
X_text = np.linspace(-5, 5, n).reshape(-1, 1)
K_ = kernel(X_text, X_test)

# Draw samples from the prior 
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
f_prior = np.dot(L, np.random.normal(size= (n, 10))) # draw 10 samples(functions)

plt.plot(X_test, f_prior)
```





















