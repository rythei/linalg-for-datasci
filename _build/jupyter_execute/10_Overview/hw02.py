# Homework 2
In this assignment, we will use the table operations that we saw in sections 2.2 and 2.4 of the workbook to investigate the `cities` dataset, which contains 251 cities in the western United States and their populations. Throughout this assignment, as in section 2.4 of the workbook, when we refer to a "function", we mean a data frame with two columns, representing the domain and the range.

We will use this table to demonstrate several properties of functions.

**NOTE: Throughout this assignment, when we ask you to "define" or "find" a function, we mean output a table with two columns representing the domain and the range of the function, as in section 2.4.**

from datasets import western_cities as cities
cities.columns = ['City', 'Population', 'State']
cities.head()

## Problem 1: Binary relations versus functions

### Part A 
Let $A =$ `State`, $B =$ `City`. Output a table representing the relation $R\subseteq A\times B$ where $\langle a, b \rangle \in R$ if $b$ is in state $a$.



Notice that this relation does not define a function from $A$ to $B$, since there are multiple cities within each state. 

### Part B 
Using group by operations, define two _different_ functions from $A =$ `State`, $B =$ `City`. Note that there will be multiple ways to do this.



## Problem 2: Images, preimages and restrictions

### Part A 
Define a function mapping each city to its population, and call this table `city_to_population`. Then compute the preimage of the set $\{x\in \mathbb{N} : x\leq 50,000\}$ under this function. What does this set represent?



### Part B
Let $S$ be the set of cities in California. Find the image of this set under the function `city_to_population`.



### Part C
Find the restriction of the function `city_to_population` to the set $S$ of cities in California.



## Problem 3: Compositions of functions
### Part A
Define function `state_to_most_pop_city` which maps each state to its most populous city. Then, compose this function with the function `city_to_population` to get the function `state_to_most_pop_city` $\circ$ `city_to_population`. What does the function you obtain represent?



### Part B
Define a _different_ function from `State` to `City` (this could be one of your functions defined in problem 1) and find its composition with the function `city_to_population`.



### Part C
Find a function `f` such that `city_to_population` $\circ$ `f` is the identity function on the set `Population`.  



### Part D
Find a subset $S$ of the set `City` such that when we restrict `city_to_population` to this subset (call this function `city_to_population_S`), `f` $\circ$ `city_to_population_S` is the identity function on the set $S$.



## Problem 4: Idempotent functions
### Part A
Find an idempotent function `g` on the set `City` which is _not_ the identity function on the set `City`. Show that this function is indeed idempotent by computing `g(City)` and `g(g(City))`.



### Part B
By restricting the function, show that your function `g` from part A is the identity function on its image.

