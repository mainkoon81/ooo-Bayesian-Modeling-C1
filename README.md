# ooo-Minkun-Model-Collection-V-
Non-parametric Bayesian Model

## Background Study
### > Gaussian Story
<img src="https://user-images.githubusercontent.com/31917400/73613995-07e46700-45f3-11ea-8760-6ae349c15dd8.png" />


### > Dirichlet Story
It's a distribution on probability distributions. It's a distribution over `n` dimensional vectors called "θ". It can be thought of as a multivariate beta distribution for a collection of probabilities (that must sum to 1). 
 - Dirichlet distribution is the conjugate prior for the **multinomial likelihood**.
 - Each `θ_i` has its own `α`...weight for each distribution of `θ_i`
 - Each `θ_i` has its own distribution.
 - Total sum of `θ_i` is 1.
<img src="https://user-images.githubusercontent.com/31917400/73609223-77daf900-45c3-11ea-97b6-52158fec1ba0.png" />

--------------------------------------------------------------------------------------------------------------------

## 1. Gaussian Process and Non-linear Problem
For any set `S`, **GP on `S`** refers to a bunch of random variables(pdf) whose index is the member of the set `S` such that they can have the following properties: The bunch of variables(pdf) are normally multivariate distributed! 

It is a distribution over functions. 
<img src="https://user-images.githubusercontent.com/31917400/73618695-60325d80-4621-11ea-8584-e57f3d37c1de.png" />






















