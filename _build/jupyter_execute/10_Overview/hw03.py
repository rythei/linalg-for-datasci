# Homework 3

In this homework, we will investigate injective and surjective functions, as well as left, right, and generalized inverses. We will do so using the table operations that we've seen in the workbook, as well as using numerical examples. These problems are primarily based on the content in section 2.6, 2.7 and 2.8 in the workbook, so please reference these as examples while completing the assignment.

## Problem 1: injective functions and left inverses with table operations

For the first two problems, we will be working with the `mountains` dataframe, which we import below:

from datasets import mountains
mountains = mountains[["Mountain", "State", "Mountain Range", "Elevation (m)"]]
mountains.columns = ["Mountain", "State", "Range", "Elevation"]
mountains.head()

This dataset contains information on 75 mountains located in the Western United States. For each mountain, the table contains information on the state it's located in, the mountain range it's located in, and its elevation.

### Part A
Output a function `f` (a table with two columns) from `Range` to `Mountain` which is injective. Verify using code that this function satisfies the properties of an injective function.



### Part B
Using the table operations covered in the workbook, construct two _distinct_ left inverses for `f`, call them `g1` and `g2` (note that these should be functions from `Mountain` to `Range`). Then, using join operations, verify by printing out the corresponding tables that `g1` $\circ$ `f` and `g2` $\circ$ `f` both form the identity function on the set `Range`.



## Problem 2: surjective functions and right inverses with table operations
### Part A
Output a function `f` (a table with two columns) from `Mountain` to `State` which is surjective. Verify that this function satisfies the properties of a surjective function.



### Part B
Using the table operations covered in the workbook, construct two _distinct_ right inverses for `f`, call them `g1` and `g2` (note that these should be functions from `State` to `Mountain`). Then, using join operations, verify by printing out the corresponding tables that `f` $\circ$ `g1` and `f` $\circ$ `g2` both form the identity function on the set `State`.



## Problem 3: an injective function
In this problem, we will consider the function $f:\mathbb{Z}\to \mathbb{R}$ (where $\mathbb{Z}$ is the set of integers and $\mathbb{R}$ is the set of real numbers) defined as follows: $f(z) = .8z + .2$ for any $z\in \mathbb{Z}$. Note that if $z, z'$ are integers with $z\neq z'$, then by definition $f(z) = .8z+.2\neq .8z'+.2 = f(z')$, and so the function $f$ is an injective function.

We define the function $f$ for your use below.

def f(z):
    return .8*z+.2

### Part A
Since the function $f$ is injective, it must have at least one left inverse $g:\mathbb{R}\to \mathbb{Z}$. Find one such left inverse, and define a python function `g(x)` implementing it.



### Part B
By plotting the function $(g\circ f)(z)$ on the integers $z\in \{-10,-9,-8,\dots,-1,0,1,\dots,8,9,10\}$, verify that it is the identity function.



### Part C
Verify that for $x$ in the set $\{-1.5, -1.25, -1.0, -.75,\dots,.75,1.0,1.25,1.5\}$, we have that $(f\circ g)$ is idempotent, namely that $(f\circ g)^2(x) = (f\circ g)(x)$.



## Problem 4: a surjective function
In this problem, we will consider the function $f:\mathbb{R}\to \mathbb{Z}$ defined as follows: for any $x\in \mathbb{R}$, $f(x) = \text{round}(x+5.1)$, where here $\text{round}$ denotes rounding to the nearest integer. As you can verify on your own, this function is surjective, since for any integer $z$ there is (at least one) real number $x$ such that $f(x) = z$. 

We define the function $f$ for your use below.

import numpy as np

def f(x):
    return np.rint(x + 5.1)

### Part A
Since the function $f$ is surjective, it must have at least one right inverse $g:\mathbb{Z}\to \mathbb{R}$. Find one such right inverse, and define a python function `g(x)` implementing it.



### Part B
By plotting the function $(f\circ g)(z)$ on the integers $z\in \{-10,-9,-8,\dots,-1,0,1,\dots,8,9,10\}$, verify that it is the identity function.



### Part C
Verify that for $x$ in the set $\{-1.5, -1.25, -1.0, -.75,\dots,.75,1.0,1.25,1.5\}$, we have that $(g\circ f)$ is idempotent, namely that $(g\circ f)^2(x) = (g\circ f)(x)$.



## Problem 5: generalized inverses for a grading problem
In this problem, we consider the following scenario. An exam is given in a course which is scored out of 100 possible points. After grading, each student receives an integer "raw" score between 0 and 100, i.e. in the set $\{0,1,2,\dots,98,99,100\}$. However, the school grading system only accepts "adjusted" scores between 59 and 100 -- students who scored below 60 on the exam are given the adjusted score of 59 to indicate a "fail" on the exam. 

### Part A
To process grades into this format, we want to use a function $f:\mathbb{Z} \to \mathbb{Z}$ which takes in the raw score, and outputs the adjusted score. Define a python function `f(z)` which implements implements this.



### Part B
The function $f$ defined above is neither injective nor surjective: it is not injective, since, for example, $f(25) = f(45)$. It is also not surjective since, for example, no integer gets mapped to the value 30. However, this function does have a generalized inverse (in fact, it has many) $g:\mathbb{Z} \to \mathbb{Z}$ satisfying $(f\circ g\circ f)(z) = f(z)$ for all $z\in \mathbb{Z}$. Find one such generalized inverse, and implement it in a python function `g(z)`.



### Part C
For integers $z\in \{0,1,2,\dots, 98,99,100\}$, verify that your function $g$ is a generalized by showing that $(f\circ g\circ f)(z) = f(z)$.



## Problem 6: more generalized inverses
In this problem, we will consider the function $f:\mathbb{R} \to \mathbb{R}$ defined as follows:
$$
f(x) = \begin{cases} 0 & \text{if } x<0\\ x^2 &\text{if } 0\leq x\leq 2\\ 4 &\text{if } x>2 \end{cases}
$$

### Part A
Define a python function `f(x)` implementing $f$ and plot it over the set `X = np.arange(-2,6.01,.01)`. 



### Part B
The function $f$ is neither injective nor surjective. It is not injective since, e.g., $f(-0.1) = f(-1000)$. It is not surjective since, e.g., there is no $x$ for which $f(x)=5$. However, it does have at least one generalized inverse $g:\mathbb{R}\to\mathbb{R}$. Find one such generalized inverse, and implement it in a python function `g(x)`. Determine the appropriate domain, and plot this function `g(x)`.



### Part C
Verify that your function $g$ is a generalized inverse for $f$ by plotting $(f\circ g\circ f)$ over the set `X` and by plotting $(g\circ f\circ g)$ over the appropriate set from Part B.



### Part D
Plot $g \circ f$ and $(g \circ f)^2$ as well as $f \circ g$ and $(f \circ g)^2$ over the appropriate sets, demonstrating that these are idempotent.

