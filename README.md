# ooo-Minkun-Model-Collection-V-
Non-parametric Bayesian Model

## Background Study
### > Gaussian Story
<img src="https://user-images.githubusercontent.com/31917400/73613995-07e46700-45f3-11ea-8760-6ae349c15dd8.png" />


### > Dirichlet Story
Dirichlet Distribution is a distribution on multinomial distributions. It is a generalized beta distribution. 
<img src="https://user-images.githubusercontent.com/31917400/73676146-c663c280-46ab-11ea-9752-f8a276cb8c20.jpg" />

 - Its **parameter `α`** will be: `n` dimensional vector which is not a pmf, but just a bunch of numbers: `c(α1, α2, α3)`  
   - if α1,α2,α3 are all the same, then the outcome(`θ_i`) appears uniformly.  
   - if α1,α2,α3 are small(<1), the outcome(`θ_i`) appears each corner and edge of the plane
     - Push the distribution to the corners.
   - if α1,α2,α3 are big(>1), the outcome(`θ_i`) appears in the center of the plane
     - Push the distribution to the middle.
   - Thus...α controls the mixture of outcomes. 
     - Turn it down, and we will likely have different values for each possible outcome. 
     - Turn it up, and we will likely have same values for each possible outcome.
 - Its **outcome `θ`** will be: `n` dimensional vector corresponding to some pmf over n possible outcomes: `c(θ_1, θ_2, θ_3) where θ_1 + θ_2 + θ_3 = 1`
It's a distribution over `n` dimensional vectors called "θ". It can be thought of as a multivariate beta distribution for a collection of probabilities (that must sum to 1). 
 - Dirichlet distribution is the conjugate prior for the **multinomial likelihood**.
 - Each `θ_i` has its own `α`...weight(scale) for each distribution of `θ_i`
 - Each `θ_i` has its own distribution...so each is a function???????
 - Total sum of `θ_i` is 1.
<img src="https://user-images.githubusercontent.com/31917400/73609223-77daf900-45c3-11ea-97b6-52158fec1ba0.png" />

### Automatic Hyperparameter Estimation? (determining parameter size??) 
 - [Q] From GMM, how to **get a control over** the latent variable(with multinomial) dynamically? We want to automatically find the **parameter**(proportions) of the latent variable at the end. 
   - The `plate notation` refers to **Random Variables** otherwise parameters. 
 <img src="https://user-images.githubusercontent.com/31917400/73740256-c57c7080-473f-11ea-8bd4-ce698ed37471.jpg" />

   - Idea 01: `Random Variablizing the **latent variable parameter**!` 
     - "latent variable" distribution = "multinomial" distribution.
     - Let's make the **parameter** of the latent variable **`"Random Variable"`** by sampling from Dirichlet(α). We can generate or vary the **parameter** for our latent variable distribution, **using Dirichlet(distribution over multinomial)** because Dirichlet is the best way to generate parameters for multinomial distribution. 
     <img src="https://user-images.githubusercontent.com/31917400/73760133-db505c80-4764-11ea-8efa-61a47729f4c7.jpg" />

   - Idea 02: `**latent variable parameter value** can be controlled by data!` but **how to address `α`?** 
     - Multinomial + Dirichlet conjugate relation tells us our parameter value(posterior) can be updated by the introduction of new data(likelihood)!
     - We can get all latent variable parameters `π` with the help of sampling `π` from Dirichlet prior! However, their occurance is not accurate? How to address the hyperparameter α that affects the sampling result ??? 
     <img src="https://user-images.githubusercontent.com/31917400/73765204-1e61fe00-476c-11ea-8bb5-3fbbb7161549.jpg" />

   - Idea 03: `**latent variable parameter value** can be.....`
     - [Note] Random Variable: RV is different from the variable in algebra as RV has whole set of values and it can take any of those randomly. Variable used in algebra cannot have more than a single value at a time: 
       - ex)`random variable_X = {0,1,2,3}`, `variable_K = 1`.
     - [Note] Random(stochastic) Process: RP is an event or experiment that has a random outcome, so you can’t predict accurately. In a deterministic process, if we know the initial condition (starting point) of a series of events, we can then predict the next step in the series. Instead, in stochastic processes, although we know the initial condition, we can’t determine with full confidence what are going to be the next steps. That’s because there are so many(or infinite!) different ways the process might evolve. Think of a stochastic process as how smoke particles collide with each other. Their unpredictable movements and collisions are random and are referred to as Brownian Motion. Interest rate is a variable that changes its value over time. It is not straightforward to forecast its movements.
       - ex) Gaussian_P, Drichlet_P, Poisson_P, Brownian motion_P, Markov decision_P,  
     
     
     
     - parameter size VS parameter value ???
       - if you know parameter size`t`, then you can expect the parameter value`w` distribution?.    
       - if you know parameter value`w`, then you can expect the data distribution?.
     - Random(Stochastic) Process refers to the infinitely labeled(infinitely hyperparameterized) collection of random variables.
     - With the passage of time(infinite hyper-parameter)`t`?? , the outcomes(resulting parameter)`w`?? of a certain experiment will change...
     <img src="https://user-images.githubusercontent.com/31917400/73839150-32aa0780-480d-11ea-9baa-3e9f4a6c712f.jpg" />

--------------------------------------------------------------------------------------------------------------------

## A. Gaussian Process and Non-linear Problem
For any set `S`, **GP on `S`** refers to a bunch of random variables(pdf functions: `f(x)`) whose index (`x`) is the member of the set `S` such that they can have the following properties: The bunch of variables(pdf functions: `f(x)`) are normally multivariate distributed, thus GP outputs from the mean function and cov function(a.k.a kernel function)!  

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



-------------------------------------------------------------------------------------------------
## B. Dirichlet Process and hyperparameter estimation???
`α` yields `π`. We want to get a control over our latent variable. The latent variable parameter `π` can be controlled by the **hyperparameter `α`** from the Dirichlet Sampling. But how are you gonna control the **hyperparameter `α`** in Dirichlet?
 - We give probability(pdf) to each hyperparameter element: (`α1`,`α2`,`α3`...)! Now we need to get a control over such probability assigning mechanism. Assuming an infinite number of hyperparameter elements,...an infinite number of multinomial values(parameters),... we can think of an infinite number of partitions - a1, a2, a3...- on the sample space.  
 - At the end of the day, the hyperparameter control(probability space partitioning to assgin to hyperparameter) can be done by managing "prior" (samples from **Dir(`α1*E[G(a1)]`,`α2*E[G(a2)]`,`α3*E[G(a3)]`...)**, then we obtain final posterior for the latent variable parameter by using the updated likelihood (which basically saying how many data pt belongs to which probability partition). 
<img src="https://user-images.githubusercontent.com/31917400/73895821-cd8bfb80-4878-11ea-9def-c2a8e3540ded.png" />





































