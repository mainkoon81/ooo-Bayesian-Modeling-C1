# ooo-Minkun-Model-Collection-V-
Non-parametric Bayesian Model
 - GP
 - DP

### Random process is a collection of random variables labeled(indexed) by `t`.  
 - When `t` variable is discrete, RP is: ![formula](https://render.githubusercontent.com/render/math?math=X_1,X_2,...X_?)
 - When `t` variable is continuous, RP is: {![formula](https://render.githubusercontent.com/render/math?math=X_t)} where t>0 
   <img src="https://user-images.githubusercontent.com/31917400/75095722-e0e4d980-558f-11ea-856e-0493d6ebb053.jpg" />

 - RP is probability distribution over `trajectories of journey of θ`(random walks) such as Markov Chain. 
   <img src="https://user-images.githubusercontent.com/31917400/75427803-12b6c100-593f-11ea-9e5e-1faf5e83b4a5.jpg" />

> ## But how Random Process can deal with infinite hyper-parameter?
parameter size VS parameter value ???
 - Run the experiments by each parameter size...denoted by `t`.  
 - If you found parameter value pool `w` by each `t`, then you can get the samples by each `t` ???
   - Random(Stochastic) Process refers to the infinite `t` (infinitely hyperparameterized) collection of random variables.
   - ## With the passage of infinite hyper-parameterization of `t` , the outcomes(resulting parameter pool)`w` of a certain experiment will change, and hit the all possible sample space. We simply want the stationary parameter pool..by summarizing all output?



## > Gaussian Story
<img src="https://user-images.githubusercontent.com/31917400/73613995-07e46700-45f3-11ea-8760-6ae349c15dd8.png" />

## > Kernel ?
<img src="https://user-images.githubusercontent.com/31917400/75116483-d2b2be00-5660-11ea-89d0-9949fac62a72.jpg" />

### Estimating a distribution?
 - Nonparametric Method(KDE): this works well when there is a lot of data without distribution knowledge.
 - Parametric Method(MLE): this works better than nonparametric when dataset is smaller and particular distribution is specified. 
 - Shrinkage Method: this works best for either high dimensional or very small datasets. It combines two weak estimators, one with high variance(Nonparametric) and the other with high bias(parametric), with some coefficient called the shrinkage coefficient(weighting), to produce a much better estimator.  

The benefit of using the Nonparametric estimator is that there are no assumptions made about the nature of the underlying distribution. So, this would give us the most general result. In Nonparametric case, the `data alone` is considered and the distribution is modeled as an **"empirical distribution"**. An empirical distribution is a distribution that has a **`kernel function at each data pt`**. This kernel function is defined to be the Dirac delta function, however, due to the difficulty of doing calculus with Dirac delta functions, modern implementations consider the "Gaussian function". If we average out kernel values from all data pt, ...this is smoothing out the empirical distribution, thus we can get our point estimate value? (but the insufficient size of data causes the nonparametric estimators to loose accuracy and become a bad estimator as stated in the "law of large numbers").

 - Using kernel? 
   - weighing neighbor points by the distance to our ![formula](https://render.githubusercontent.com/render/math?math=x_i). The point that are really neighbor to our ![formula](https://render.githubusercontent.com/render/math?math=x_i) have the higher weight and the points that are far away have lower weights.
   - The weight can be computed as the **kernel function** of "x" (`the x value for the point(y) that we want to predict`) and ![formula](https://render.githubusercontent.com/render/math?math=x_i) (the position of the `i`th neighbor point). 
   - In Gaussian kernel, if we take a higher value of sigma, the values would drop slower obviously, so you can weight further points a bit higher...getting closer to Uniform?  If sigma is low then the kernel would quickly drop to zero, so each point can get extreme weight. 
   - In Uniform kernel, we equally weigh the points. 
   <img src="https://user-images.githubusercontent.com/31917400/75113588-1ac4e700-5647-11ea-9ccd-577e091f44b2.jpg" />

### Why use kernel ??????????
 - classification(svm): creating customized feature space
 - regression(knn): weighting neighboring points
 - otheres: smoothing histogram(kde),...giving customized samples(behaving as the covariance) in the random process
 - a kernel is a weighting function..by calculating each kernel value for each data point. 
 - ??????????????????????????

--------------------------------------------------------------------------------------------------------------------
# A. Gaussian Process and Non-linear Problem
Before using Gaussian Processes, you have two main questions: 
 - Q1. "Is your dataset reasonably small?" (less than 1,000 is fine)
   - Gaussian Processes don't do well with large numbers of points.
 - Q2. "Do you have a good idea about how two point's labels are related?" 
   - ex) if you know f(1) = A, do you have any good intuition about what that makes you think f(4) is? This is crucial for picking a reasonable covariance function. It’s useful when you want a measure of uncertainty about your predictions instead of just a point estimate. "What’s the probability that f(4) is between 2 and 7?"  then a Gaussian process with a linear-kernel would make sense.

**GP** on `S` refers to a bunch of random variables(with pdf: `f(x)`) on `S`... whose index is the member of the set `S` such that they can have the following properties: The bunch of variables ![formula](https://render.githubusercontent.com/render/math?math=x_n) with pdf functions: f(![formula](https://render.githubusercontent.com/render/math?math=x_n)) are normally multivariate distributed, thus GP outputs from the **mean function** and **cov function (kernel function)**! Gaussian process is parameterized by the `mean vector` and the `covariance matrix`.
<img src="https://user-images.githubusercontent.com/31917400/76784936-7731a700-67ac-11ea-8170-d25a73254c94.jpg" /> 

Kernel helps us obtain customized samples in the random process. And if we keep increasing the value of bandwidth in the kernel, we'll have almost **constant functions**...weighted evenly???? 
<img src="https://user-images.githubusercontent.com/31917400/75114699-439daa00-5650-11ea-962d-608bb877a093.jpg" /> 

> ## But for what do we need such customized(weighted) samples? 

### 1> GP Regression
<img src="https://user-images.githubusercontent.com/31917400/76789020-ea8ae700-67b3-11ea-8ea5-a0c3175a7688.jpg" /> 

**GP** is a distribution over functions. 
<img src="https://user-images.githubusercontent.com/31917400/73618695-60325d80-4621-11ea-8584-e57f3d37c1de.png" />

To make the prediction, take the point (x, f(x)), then try to generate the mean and CI(or cov): Given that having training set, I try to combine the data with the prior of the functions(with mean and cov)..to make my functions smooth. Use a similarity matrix such that our function approximator makes the two points close by when we fit new data to make sure the two heights are also close by.    
<img src="https://user-images.githubusercontent.com/31917400/73652179-2d6a8280-467e-11ea-8651-2c115fe2c1e7.png" />

### 2> GP Classification


### 3> Some tricks for GP training
 - Case 01> Too noisy observations:
   - Add the **independent Gaussian noise** to all RVs.
     - we don't have the `0 variance` data points anymore. And also the **mean function became a bit smoother**.
     <img src="https://user-images.githubusercontent.com/31917400/76793814-a6044900-67bd-11ea-9e1a-bbd5e017c0c9.jpg" /> 

   - Hey, we can change the parameters(bandwidth of kernel, variance of GP, variance of noise) of the kernel a bit, and find the optimal values for them, in this special case. 
     - Parameter Optimization using Maximum Likelihood Estimation  
     <img src="https://user-images.githubusercontent.com/31917400/76794907-f9779680-67bf-11ea-96d0-2982a05dd48b.jpg" /> 


 - Case 02> Inducing Input













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











## > Dirichlet Story
### Beta eats `α`, `β` and spits out `θ`.
 - `α`, `β` are shape parameters.
 - **Beta(`α`, `β`) prior** takes Bin(n, `θ`) likelihood
### Dirichlet eats `α1`,`α2`,`α3`... and spits out `θ1`,`θ2`,`θ3`,...
 - Dirichlet Distribution is a generalized beta distribution.
 - `α1`,`α2`,`α3`... are shape parameters.
 - **Dirichlet(`α1`,`α2`,`α3`...) prior** takes Multinom(n, `θ1`,`θ2`,`θ3`...) likelihood.
 <img src="https://user-images.githubusercontent.com/31917400/169688590-af305404-bf2b-41bc-9f34-37fc012d44c6.jpg" />

 - Its **parameter `α`** will be a shape vector.  
   - if α1,α2,α3 are all the same, then the outcome(`θ_i`) appears uniformly.  
   - if α1,α2,α3 are large(>1), the outcome(`θ_i`) appears in the center of the plane (convexed) 
   - if α1,α2,α3 are small(<1), the outcome(`θ_i`) appears each corner and edge of the plane (concaved)
   - Thus...α controls the mixture of outcomes. 
     - Turn it down, and we will likely have different values for each possible outcome. 
     - Turn it up, and we will likely have same values for each possible outcome.
    
 - Its **outcome `θ`** will be `k` dimensional vector such as `c(θ_1, θ_2, θ_3) where θ_1 + θ_2 + θ_3 = 1`
   - Each `θ_i` has its own `α`...weight(shape) for each distribution of `θ_i`
   - Each `θ_i` has its own distribution...???????

## What Components ? 
Gaussian Mixture example
 - How to **get a control over** the `multiple membership`(?) dynamically? 
 - Note that the `plate notation` refers to **Random Variables** (`multiple membership`). 
 <img src="https://user-images.githubusercontent.com/31917400/169689237-647c1da7-ce61-47c7-9ba1-8e5c9280184a.jpg" />

   - ### The membership pool "Z" can be expressed in two ways: 
     - 1.Collection of the cluster parameters `μ` and `Σ` 
     - 2.Collection of the cluster proportions `π`
       - Here, just focus on **π**: obv-proportions for the membership "Z". 
       - ![formula](https://render.githubusercontent.com/render/math?math=\pi_i=P(\mu_i,\Sigma_i))
       - In Gaussian Mixture, ![formula](https://render.githubusercontent.com/render/math?math=Z=(\pi_1,\pi_2,...)). It's a random clustering case..so it can vary! Z, Z, Z, Z,...all different..
       - ![formula](https://render.githubusercontent.com/render/math?math=\pi_i=\theta_i)
       - In Dirichlet Mixture, ![formula](https://render.githubusercontent.com/render/math?math=Z=(\theta_1,\theta_2,...)). It's a random clustering case..so it can vary! Z, Z, Z, Z,...all different..     
     - Gaussian Mixture vs Dirichlet Mixture
       - ![formula](https://render.githubusercontent.com/render/math?math=y_1,y_2,...y_i~MVN(Z_k)): `Likelihood` (This is for Gaussian Mixture)
       - ![formula](https://render.githubusercontent.com/render/math?math=Z_k=((\mu_1,\Sigma_1),(\mu_2,\Sigma_2),..)~normalInverseWishart(\mu_0,\kappa_0,\nu_0,\nu_0\Sigma_0)): `prior` (This is for Gaussian Mixture)
       - ![formula](https://render.githubusercontent.com/render/math?math=y_1,y_2,...y_i~Multinom(Z_k)) : `Likelihood` (This is for Dirichlet Mixture) 
       - ![formula](https://render.githubusercontent.com/render/math?math=Z_k=(\theta_1,\theta_2,...\theta_k)~Dirichlet(\alpha_1,\alpha_2,..\alpha_k)) : `Prior` (This is for Dirichlet Mixture) 
       - Multinomial + Dirichlet conjugate relation tells us our parameter value(posterior) can be updated by the introduction of new data(likelihood)! We can get all latent variable with the help of sampling `θ` from Dirichlet prior! So it seems we can easily get the posterior, thus our `θ` ("mixing coefficients" or "obv-proportion" for every Gaussians) at the end. Done and dusted! We now have the model describing our data! Wait! However, is their occurance accurate? How to address the hyperparameter α that affects the sampling result ?
       <img src="https://user-images.githubusercontent.com/31917400/169690028-97deca2d-240c-4866-9005-9d7e1ed5f039.jpg" />
     
   - ### How are you gonna deal with **`α`** and what if `k` goes to infinity? 
   
[Idea]: `**infinite latent variable parameter values** can be controlled by Random Process that can address **α**`
> Note: Random Variable & Random Process
 - : RV is different from the variable in algebra as RV has whole set of values and it can take any of those randomly. Variable used in algebra cannot have more than a single value at a time: 
   - ex)`random variable_X = {0,1,2,3}`, `variable_K = 1`.
 - : Random(stochastic) Process: Random Process is an event or experiment that has a random outcome, so you can’t predict accurately. In a deterministic process, if we know the initial condition (starting point) of a series of events, we can then predict the next step in the series. Instead, in stochastic processes, although we know the initial condition, we **can't determine with full confidence** what are going to be the next steps. That’s because there are so many(or infinite!) different ways the process might evolve. How smoke particles collide with each other? Their unpredictable movements and collisions are random and are referred to as Brownian Motion. **Interest rate is a variable that changes its value over time. It is not straightforward to forecast its movements.** - ex) Gaussian_P, Drichlet_P, Poisson_P, Brownian motion_P, Markov decision_P, etc... Markov Chain is also random process(resulting random ouput) in which the effect of the past on the future is only summarized by the current state.  

-------------------------------------------------------------------------------------------------
# B. Dirichlet Process and `α`, `G_0`
First, assume you have data that follows some **unknown mixture distribution**. so we want to estimate `mixing coefficeint`(proportion), and other distribution specific certain `parameters` for each **mixture components** (cluster). For the time being, forget about the labelling the membership because if K goes to infinity, our "Z case" (Scenarios) becomes random variable. Now it's time for Random Process! Ok...What on earth is DP ? It seems both DD, DP are `distributions of certain scenarios`. <img src="https://user-images.githubusercontent.com/31917400/170220340-a18782a0-4bf0-4bb4-8e98-11b3079abd82.jpg" />

- The **G_0** is the `joint distribution of the parameters for a potential new cluster` for an observation to enter. The new cluster is initially empty, so there is no data that could be used to determine the **posterior estimate of the parameters**. Hence, we instead draw parameters from the **prior distribution G_0** to determine estimates for the parameters, which then get used in the calculations of the probability of entering that cluster. If a new cluster is indeed selected, then the **G_0 is discarded** and a new `δ_Φ` is created for a brand new cluster. Then this new `δ_Φ` is used as the distribution for probability calculations, parameter updates, etc, for future observations that may want to enter that cluster. **G_0** would be used again for another new cluster if necessary. **G_0** is like a real estate aganet. It keeps suggesting new cluster locations to the data point. Then the data point decides if staying or moving out based on the porbability built by the proportion and likelihood of its `current cluster` vs `new cluster`. Ok. the **importance weight**(proportion?) and **cluster parameter**(location?) information are the key criteria from which your data is understood.

- **Q.** then...the initial importance weight for **G_0** is determined by **`α`**? **`α`** is integer values sampled from some algorithm? 
- **Q.** `α` controls the concentration around the mean (smaller α makes the concentration tighter since it gives smaller variance). This means `α` is analogous to the `variance of sampling`. Note, in Beta(α,β), the smaller shape gives bigger probability...so it makes sense. 
- **Q.** However, a larger `α` allows more clusters? and a larger `α` give higher proportion to **G_O**? How to explain this? Can we say a latent variable = `cluster proportion`? so..can we say the latent variable **θ**(`cluster proportion`) can be controlled by the `α`? 
- **Q.** G_0 is a joint prior distribution. `Φ` is a random variable of G_0. G_0(A) is AUC. **`α`** affects AUC? AUC are not importance weight? which stage the partition of AUC occurs? and what samples the AUC? **A** indicates the cluster index? G_0(A) is AUC and it is also Expected value of G(A)? then..it's not about the cluster proportion??
- **Q.** **G_0(A)** vs **G(A)** ?? `prior` vs `predictive distribution` that are equipped with posterior parameter and new likelihood data entering. 
- **Q.** why the output **G** is denoted with ``Summation"? why not a collection of `δ_Φ` ? The summation implies a solid, stand-alone scenario (Mixture Model)? so... G(A) is a component of G? so at the end...G is again...a predictive distribution?



### What is DP?
For any partition ![formula](https://render.githubusercontent.com/render/math?math=A_1,...,A_K) of the support of `G0`, of any size `K`, the **collection of cluster information**(parameters): (![formula](https://render.githubusercontent.com/render/math?math=G(A_1),...,G(A_K))) follows a **Dirichlet distribution** with k-dimensional DD parameters ![formula](https://render.githubusercontent.com/render/math?math=\alpha*G_0(A_1),...,\alpha*G_0(A_K)). In other words, DP can sample all possible highly likely **`scenarios of mixture setup`** (mixture models ???) that describes your complex data.  

### Clustering example
<img src="https://user-images.githubusercontent.com/31917400/170001808-e773779e-adad-4ce2-aaeb-10cbeb142400.jpg" />



### How to decide the membership of new data pt? 
At the end of the day, the constructing(estimating) cluster is done by sampling. For sampling, we need the DP prior, i.e. We want to sample the "function" **G**(distribution over proportions of each `θ` as a random "clustering scheme") from prior: DP(`α`, `G0`). 
<img src="https://user-images.githubusercontent.com/31917400/75469279-a495ec80-5986-11ea-8a39-f66d176a9270.jpg" />

We can construct the DP prior, using Non-parametric **prior construction** scheme, then assign a membership to new data.  
 - Sol 1) Stick-Breaking scheme(**creating "G" distribution**)
   - : Big `α` results in big sticks while small `α` results in small sticks.
 - Sol 2) Chinese-Restaurant scheme(**assigning membership to new point**)
   - : A customer is more likely to sit at a table if there are already many people sitting there. However, with probability proportional to `α`, the customer will ask a new table.
     
 - ### [1] Stick-Breaking scheme: 
   - **`Creating a decent distribution: G(A_i)`**, "the each single stick", an element of **G**. 
   - How to obtain a candidate probability values of the pizza with infinite slicing?
     - Using the "adjusted Beta value": **GEM(hyperparameter `α`)** which is an adjusted probability value. 
     - Based on the properties of Beta:
       - Big Hyperparameter: result in big sticks
       - Small Hyperparameter: result in small stick
   <img src="https://user-images.githubusercontent.com/31917400/74085265-33da6f00-4a6f-11ea-9daa-2625a3e15f0b.jpg" />
  
 - ### [2] Chinese-Restaurant-Process scheme:
   - **`Assigning a membership to new point`**
   - here, the number of tables(and hence table parameters θ) will grow with N, which does not occur in a finite model. This is the essence of the "non-parametric" aspect.
   - Rich get richer...  popular table..
   - No fixed size of labels with a fixed size of data instances
   <img src="https://user-images.githubusercontent.com/31917400/74458192-3ed33c00-4e81-11ea-928c-8d06879909de.jpg" />

### Inference:
The main goal of clustering is to find the posterior distribution **P(![formula](https://render.githubusercontent.com/render/math?math=\pi_n)|x)** of the cluster assignments! Computing this is intractable due to the sum in the denominator and the growing number of partitions. That's why we use Gibbs Sampling. Let's say..given the previous partition ![formula](https://render.githubusercontent.com/render/math?math=\pi_n), we remove one data pt `x` from the partition (prior) then re-added to the partition (likelihood) to obtain posterior: **P(![formula](https://render.githubusercontent.com/render/math?math=\pi_n)|`x`)**. This gives **new partition** (prior)! 
 - de Finetti's theorem + GibbsSampling
   - The exechangeability is important coz...
     - Chinese-Restaurant-Process is exchangeable process
     - Gibbs Sampling should use the exchangeability coz...its sampling is carried out one label by one label...so can ignore labeling order. 
     <img src="https://user-images.githubusercontent.com/31917400/74452857-86ee6080-4e79-11ea-8676-5b0357881917.jpg" />

### DP Mixture Model:   
**G** from DP is `discrete` with probability "1", thus DP would not be a suitable prior distribution for the situations with continuous data coz in this case, we want continuous **G**. Let's think about mixture models. Mixture models are widely used for **density estimation** and classification problem. A natural idea is to create a prior for `continuous` densities via a mixture where the mixing distribution **G** is given a Dirichlet process prior. As a natural way to increase the applicability of DP-based modeling, we can use DP as a prior for the mixing distribution in a mixture model with a `parametric kernel distribution`. 

The posterior under a DPMM is effectively finite-dimensional, though the dimension is adaptive, determined by data, as opposed to fixed like in the parametric models. This **adaptive dimensionality** is what gives the model its flexibility and its effective **finite-dimensionality** is what makes posterior computation possible.

DP gives a `cdf` while DPMixture gives a `density` ???? 
 - One hurdle we encounter here is **"sampling from `G`"**, which has countably many atoms(sticks). There is also an exact approach that generates atoms "on the fly" as needed, and exploits the fact that only finitely many atoms are needed in practice.  
 <img src="https://user-images.githubusercontent.com/31917400/75472976-479d3500-598c-11ea-8f4e-2f6516b9b374.jpg" />

 - For the sampling algorithm, it is convenient to include table assignment variable `Z` to indicate which table our data pt ![formula](https://render.githubusercontent.com/render/math?math=\x_i) belongs to.  
 <img src="https://user-images.githubusercontent.com/31917400/74239298-7ac69f80-4ccf-11ea-9474-e0eb493dee18.jpg" /> where `Cat()` refers to a categorical or multinoulli distribution? 

 - We have the joint so perform Gibbs sampling over the state space of {`w`, `Փ`} and {`z`} <img src="https://user-images.githubusercontent.com/31917400/74241108-4359f200-4cd3-11ea-8621-611ed9657be7.jpg" /> At each iteration, we choose one of these variables and re-sample it from its conditional distribution given all the other variables.

Our Dirichlet Process provides a discrete distribution over objects and take i.i.d. samples from this distribution. Analogous to the `beta-binomial` and `Dirichlet-multinomial` conjugacy, we suspect the **posterior of the DP**, after observe samples, is also a DP. We will make this precise: 
 - Suppose we have a partition ![formula](https://render.githubusercontent.com/render/math?math=A_1,A_2,...A_K).
 - The vector (![formula](https://render.githubusercontent.com/render/math?math=\delta\theta_i(A_1),...\delta\theta_i(A_K))) is an indicator vector for the index `k` such that ![formula](https://render.githubusercontent.com/render/math?math=\theta_i\in%20A_k). And this event (conditioned on G) has probability G(![formula](https://render.githubusercontent.com/render/math?math=A_k)).
   - so ![formula](https://render.githubusercontent.com/render/math?math=A_1,A_2,...A_K) are K different world, and for each world, same ![formula](https://render.githubusercontent.com/render/math?math=\theta_i) can exist with different probability: ![formula](https://render.githubusercontent.com/render/math?math=\delta\theta_i(A_k)). ![formula](https://render.githubusercontent.com/render/math?math=\theta_i) is drawn from **G**. ????? 
 - Thus, this vector(conditioned on G) (![formula](https://render.githubusercontent.com/render/math?math=\delta\theta_i(A_1),...\delta\theta_i(A_K))| G) is a categorical/multinoulli random variable with parameters ![formula](https://render.githubusercontent.com/render/math?math=G(A_1),G(A_2),...G(A_K)). 
 - And these random parameters ![formula](https://render.githubusercontent.com/render/math?math=G(A_1),G(A_2),...G(A_K)) follow a Dirichlet distribution! This is the primary reason for the name "Dirichlet Process". 
 <img src="https://user-images.githubusercontent.com/31917400/74276686-170f9700-4d0e-11ea-8445-f4fa64978f99.jpg" />

 - Since the Dirichlet distribution is conjugate to the multinomial distribution, we can have the posterior!
 <img src="https://user-images.githubusercontent.com/31917400/74277676-dc0e6300-4d0f-11ea-9538-3422cf4ef44d.jpg" />


### > Implementation
 - Step I. Initial Labeling(assign table) to every point.
 - Step II. While Gibbs Sampling Iteration with each data instance:
   - For each data instance:
     - Remove the instance from the label
     - Calculate the prior labal 
     - Calculate the likelihood 
     - Calculate the posterior
     - Sample the label from the posterior
     - Update the component parameter


































