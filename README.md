# ooo-Minkun-Bayesian-Modeling-C1
Non-parametric Bayesian Model
 - GP
 - DP

### Random process is a collection of random variables, defined on a common probability space, taking values in a common set S (the state space), and indexed (labeled) by a set T, thought of as time (discrete or continuous respectively).
 - When `t` variable is discrete, RP is: ![formula](https://render.githubusercontent.com/render/math?math=X_1,X_2,...X_?)
 - When `t` variable is continuous, RP is: {![formula](https://render.githubusercontent.com/render/math?math=X_t)} where t>0 
 <img src="https://user-images.githubusercontent.com/31917400/75095722-e0e4d980-558f-11ea-856e-0493d6ebb053.jpg" width="60%" height="60%" />

 - RP is probability distribution over `trajectories of journey of θ`(random walks) such as Markov Chain. 
 <img src="https://user-images.githubusercontent.com/31917400/75427803-12b6c100-593f-11ea-9e5e-1faf5e83b4a5.jpg" width="80%" height="80%" />

> ## But how Random Process can deal with infinite hyper-parameter?
parameter size VS parameter value ???
 - Run the experiments by each parameter size...denoted by `t`.  
 - If you found parameter value pool `w` by each `t`, then you can get the samples by each `t` ???
   - Random(Stochastic) Process refers to the infinite `t` (infinitely hyperparameterized) collection of random variables.
   - ## With the passage of infinite hyper-parameterization of `t` , the outcomes(resulting parameter pool)`w` of a certain experiment will change, and hit the all possible sample space. We simply want the stationary parameter pool..by summarizing all output?



## > Gaussian Story
<img src="https://user-images.githubusercontent.com/31917400/73613995-07e46700-45f3-11ea-8760-6ae349c15dd8.png" />

--------------------------------------------------------------------------------------------------------------------
# A. Gaussian Process and Non-linear Problem
Before using Gaussian Processes, you have two main questions: 
 - Q1. "Is your dataset reasonably small?" (less than 1,000 is fine)
   - Gaussian Processes don't do well with large numbers of points.
 - Q2. "Do you have a good idea about how two point's labels are related?" 
   - ex) if you know f(1) = A, do you have any good intuition about what that makes you think f(4) is? This is crucial for picking a reasonable covariance function. It’s useful when you want a measure of uncertainty about your predictions instead of just a point estimate. "What’s the probability that f(4) is between 2 and 7?"  then a Gaussian process with a linear-kernel would make sense.

**GP** on `S` refers to a bunch of random variables(with pdf: `f(x)`) on `S`... whose index is the member of the set `S` such that they can have the following properties: The bunch of variables ![formula](https://render.githubusercontent.com/render/math?math=x_n) with pdf functions: f(![formula](https://render.githubusercontent.com/render/math?math=x_n)) are normally multivariate distributed, thus GP outputs from the **mean function** and **cov function (kernel function)**! Gaussian process is parameterized by the `mean vector` and the `covariance matrix`.

<img src="https://user-images.githubusercontent.com/31917400/76784936-7731a700-67ac-11ea-8170-d25a73254c94.jpg" width="80%" height="80%" /> 

Kernel helps us obtain customized samples in the random process. And if we keep increasing the value of bandwidth in the kernel, we'll have almost **constant functions**...weighted evenly???? 

<img src="https://user-images.githubusercontent.com/31917400/75114699-439daa00-5650-11ea-962d-608bb877a093.jpg" width="60%" height="60%" /> 

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
     <img src="https://user-images.githubusercontent.com/31917400/76793814-a6044900-67bd-11ea-9e1a-bbd5e017c0c9.jpg" width="60%" height="60%" /> 

   - Hey, we can change the parameters(bandwidth of kernel, variance of GP, variance of noise) of the kernel a bit, and find the optimal values for them, in this special case. 
     - Parameter Optimization using Maximum Likelihood Estimation  
     <img src="https://user-images.githubusercontent.com/31917400/76794907-f9779680-67bf-11ea-96d0-2982a05dd48b.jpg" width="60%" height="60%" /> 


 - Case 02> Inducing Input
   - T.B.D

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
 <img src="https://user-images.githubusercontent.com/31917400/169688590-af305404-bf2b-41bc-9f34-37fc012d44c6.jpg" width="60%" height="60%" />

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
 <img src="https://user-images.githubusercontent.com/31917400/169689237-647c1da7-ce61-47c7-9ba1-8e5c9280184a.jpg" width="60%" height="60%"/>

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
# B. Dirichlet Process Outcome Modelling Framework and `α`, `G_0`
A **collection of cluster information**(parameters): (![formula](https://render.githubusercontent.com/render/math?math=G(A_1),...,G(A_K))) follows a **Dirichlet distribution** with k-dimensional DD parameters ![formula](https://render.githubusercontent.com/render/math?math=\alpha*G_0(A_1),...,\alpha*G_0(A_K)). DP can sample all possible highly likely **`scenarios of mixture setup`** (mixture of clusters) that describes your complex data. First, assume you have data that follows some **unknown mixture distribution**. so we want to estimate `mixing coefficeint`(proportion), and other distribution specific certain `parameters` for each **mixture components** (cluster). For the time being, forget about the labelling the membership because if K goes to infinity, our "Z case" (Scenarios) becomes random variable. Now it's time for Random Process! Ok...What on earth is DP ? It seems both DD, DP are `distributions of certain scenarios`. <img src="https://user-images.githubusercontent.com/31917400/175239736-26795abe-aeb5-41f1-978d-6004e70e1db6.jpg" />

- **Remarks:** The **G_0** is the `joint distribution of the parameters for a potential new cluster` for an observation to enter. The new cluster is initially empty, so there is no data that could be used to determine the **posterior estimate of the parameters**. Hence, we instead draw parameters from the **prior distribution G_0** to determine estimates for the parameters, which then get used in the calculations of the probability of entering that cluster. If a new cluster is indeed selected, then the **G_0 is discarded** and a new `δ_Φ` is created for a brand new cluster. Then this new `δ_Φ` is used as the distribution for probability calculations, parameter updates, etc, for future observations that may want to enter that cluster. **G_0** would be used again for another new cluster if necessary. **G_0** is like a real estate aganet. It keeps suggesting new cluster locations to the data point. Then the data point decides if staying or moving out based on the **porbability built by the `proportion` and `likelihood` of its current cluster vs new cluster**. Ok. the **importance weight**(proportion?) and **cluster parameter**(location?) information are the key criteria from which your data is understood.

- **Remarks:** Then...the initial importance weight for **G_0** is determined by **`α`**? **`α`** is a value sampled from some algorithm? And it's like a reliability of **G_0**'s suggestion. The higher **`α`** gives higher proportion of the new cluster, thus, data might choose the new cluster more often. `α` controls the concentration around the mean (smaller α makes the concentration tighter since it gives smaller variance, thus produce less clusters). This means `α` is analogous to the `variance of sampling`. Note, in Beta(α,β), the smaller shape gives small variance, thus produces bigger probability (peak)...so it makes sense. Hence, a larger `α` allows more clusters and a larger `α` give higher proportion to **G_O**'s suggestion.

- **Remarks:** **G_0** is a joint prior distribution. `Φ` is a mixed random variable of **G_0**. `G_0(A)` is AUC, but it is not an importance weight (cluster proportion). AUC is not important. It's just a representation. **G_0(A)** vs **G(A)** :: `prior parameter samples` vs `posterior cluster density` that are equipped with posterior parameter and new likelihood data entering. **G(A)** is a component of **G** and **G** as a single output of Dirichlet process can be represented in two ways: 1) a collection of `proportion x cluster densities(δ_Φ)`, 2) a solid, single, stand-alone predictive Mixture Model using the weighted AVG. 


# [Note]
The following three steps make one iteration. 
- [Step 1] **Clustering** (Accept/Reject G_0's suggestion) 
  - The unique cluster proportions or the probability of each cluster (`π1`,`π2`,`π3`...`πk`) are defined as it goes...then the possibility of new cluster probability `πk+1` are considered when we assign new observation to the brandnew clusters (`parameter free joint` of Y and X) and data pt decide to move in or not based on the probability comparison. 
  - In this stage, **cluster memberships**(k) are fully defined. 

- [Step 2] **Outcome_Model, Covariate_Model Refinement**  
  - This is for the `parameter update`, the `unique cluster probability term development` later.
  - Given the data points in each clusters, some extra strategies (such as imputation for missing data, etc.) can be implemented to refine them.  
  - For Outcome Model, we filling in any missing data, then integrate them out the covariates with missing data. So..it looks like the model ignores the missing data, but it doesn't.. 
  - For Covariate Model, we naturally ignore any covariates with missing data. 

- [Step 3] **Parameter Re-Estimation** (**Posterior** development based on the likelihood)
  - Note that in the parameter estimation stage, you need to have a **complete data!** 
  - The prameters of each cluster (X`β`, `σ^2`) are re-calculated based on the data pt they have.. 

# > Detail I. IntraCorrelation strategy 
<img src="https://user-images.githubusercontent.com/31917400/173237731-f3150a14-c6c6-48c5-b04a-864c0542b3a3.jpg" width="90%" height="90%" />

# > Detail II. missing Data strategy
<img src="https://user-images.githubusercontent.com/31917400/176428959-3b27b335-4f96-4c4c-b785-a16106c51268.jpg" width="70%" height="70%" />

Once all missing data in all covariates has been imputed, then the prameters of each cluster (X`β`, `σ^2`) are re-calculated. After this parameter has been updated, the clustering process is performed and in the parameter Re-estimation stage, the previous imputed data is discarded and the sampling for the imputation starts over in the next iteration. This means... `missing data do not impact on the clustering process whatsoever in the iteration`.  Aslo note that when calculating the predictive distribution, we integrate out the covariates that are missing, thus remove these two missing covariates from the `x` term that is being conditioned on: `p(y|X, θ)` => `p(y|x1,x4,θ)`. 

# > Detail III. measurement Error strategy
## - Mismeasured Continuous Predictor
### - 1) Base Method
The goal is to find the distribution of the clean covariate!  
<img src="https://user-images.githubusercontent.com/31917400/176433381-54d4de40-aaf1-48e2-92b5-2418841caa45.jpg" width="70%" height="70%" />




### - 2) DP Incorporation




## - Mismeasured Binary Predictor
t.b.d....





# Story on `α`
## Classic
Escobar and West developed the posterior distribution for the DP prior: `α` as follows:
<img src="https://user-images.githubusercontent.com/31917400/171143039-e3e1326a-0b80-408c-9345-d526c0538520.jpg" />

## Others
We can construct the DP prior: `α` (Suggestion Reliability), using Non-parametric **prior construction** scheme. 
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


































