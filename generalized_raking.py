import numpy as np
import math
from numba import njit
from scipy.optimize import minimize



@njit(fastmath=True)
def fast_fun(x):
    s = 0.
    for i in range(x.shape[0]):
        s+=(x[i]*(math.log(x[i])-1) + 1 )**2

    return math.sqrt(s)

# in this case Numba can parallelize some of the operations
@njit(fastmath=True, parallel=True)
def fast_constr(w,x,b):
    c = np.dot(x.T,w)-b
    s = 0.
    for i in range(c.shape[0]):
        s+=(c[i])**2
        
    return math.sqrt(s)



def gen_raking(data_classes, pop_shares, ftol=1e-2, maxiter=500, eps=1e-8, verbose=False):
    """Function generating sample weights (typically for survey data) using generalized 
    raking to match the population shares for multiple characteristics. Shares are matched
    independently and not as crossproducts. As opposed to Iterative Proportional Fitting, 
    the constrained optimization problem is solved, where the objective is to generate weights
    as close to 1 as possible: minimizing w(log(w) - 1) + 1, as suggested in Deville et al. (1992)
     to avoid having samples with very high/low weights.

    Args:
        data_classes (numpy.ndarray): Array of shape n x k, where n is number of samples and 
                                    k number of characteristics we use to match the population
                                    distribution. The i-th row represents the numbers of groups the i-th
                                    sample is member of. All the individual characteristics in the 
                                    population need to be represented by some sample in the data.
        pop_shares (list): list of length k (number of characteristics used to match the population) 
                            containing lists. Every individual list contains shares of the allowed 
                            values of the individual characteristic in the target population.

        ftol (float): Precision goal for the value of f in the stopping criterion.
        maxiter (int): Maximum number of iterations for the optimization algorithm.
        eps (float): Step size used for numerical approximation of the Jacobian.
        verbose (bool): whether to print the target distributions and distributions of weighted data.


    Returns:
        _type_: _description_
    """

    # we want to check that we have all and only the target groups represented in the data
    for i in range(data_classes.shape[1]):
        if set(np.unique(data_classes[:,i])) != set(range(len(pop_shares[i]))):
            raise ValueError("""Characteristics in the data do not match the population 
            characteristics. Either one group not present in the data or too many groups 
            present (characteristic: {})".format(i+1)""")

    # adjust shares to actual counts of samples we want to have in every group
    for i in range(data_classes.shape[1]):
        pop_shares[i] = pop_shares[i]*data_classes.shape[0]/pop_shares[i].sum()

    # characteristics of the data must be recoded into dummies, separate array for every variable
    types = []
    for i in range(data_classes.shape[1]):
        values = data_classes[:,i].astype("int")
        n_values = np.max(values) + 1
        types.append(np.eye(n_values)[values])


    # optimization constrains in the form X'W = b. ie weighted samples need to represent defined population
    cons = []
    for i in range(data_classes.shape[1]):
        cons.append({"type":"eq", "fun": lambda w, x = types[i], 
                    b = pop_shares[i] : fast_constr(w,x,b)})

    # initial weights - let's start with naive 1's
    x0 = np.ones(data_classes.shape[0])
    # bounds for our weights - we don't want 0 or negative weights
    b = [(0.001,None)]*data_classes.shape[0]
    
    # 1.5 Version of scipy throws error here. It should work with 1.4.1 or with new versions
    res = minimize(fast_fun, x0 = x0, constraints=cons, bounds = b, method='SLSQP', 
                    options=dict(ftol=ftol, maxiter=maxiter, eps=eps, disp=False) ) 

    weights = res.x

    if verbose:
        print("Comparison of target and shares in weighted data:")
        for i in range(data_classes.shape[1]):
            print("Target {}".format(i))
            print(pop_shares[i])
            print("Weighted data:")
            print(np.dot(types[i].T,weights))


    return weights


# Simple example:

# generating age (5 possible levels) and education level (4 levels) 50 observations
age = np.random.randint(0,5,50)
educ = np.random.randint(0,4,50)

# numpy array where one row = one observation, 
# every column representing one characteristic we use for weighting
to_weight = np.column_stack((age, educ))

# preparing array for the weights
weights = np.zeros(to_weight.shape[0])

# defining shares in population
age_pop = np.array([0.2, 0.25, 0.15, 0.3, 0.1])
educ_pop = np.array([0.25, 0.25, 0.4, 0.1])

# getting respective counts in the population of size of our survey sample (50 observations)
population = []
population.append(age_pop*to_weight.shape[0])
population.append(educ_pop*to_weight.shape[0])

weights = gen_raking(to_weight, population, ftol=0.001, maxiter=500, eps=1e-8, verbose=True)