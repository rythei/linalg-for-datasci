#!/usr/bin/env python
# coding: utf-8

# # Taking Powers of Matrices
# 
# Taking powers of matrices refers to multiplying a matrix by itself multiple times.
# 
# Given an $n \times n$ matrix $A$, we can multiply it by itself to get $ AA $, and this can be denoted $A^2$. 
# Then, since $A^2$ is an $n \times n$ matrix, we can multiply it to get $A^3$, and so on 
# 
# $$
# A^k = A \cdots A  ,
# $$
# 
# where there are $k$ copies of $A$ in this product.
# Taking powers of matrices is very common.
# Here, we will explore some of the properties of this operation.
# 
# There are two (related) ways to view this.
# - First, we can simply multiply $A$ by itself multiple times and see what happens.
# - Second, we can think of $A$ as multiplying a vector $x$, as in $Ax$, and see what happens if we do this multiple times, i.e., what is $A(A(\cdots(A(x))))$.
# 
# We are interested in both of these views.
# 
# 
# ## Powers of a diagonal matrix 
# 
# Let's start by considering powers of a diagonal matrix.
# 
# Consider a diagonal matrix $A$ of the form
# 
# $$
# A = \begin{pmatrix}a_1 & 0 &\cdots &0\\
# 									 0   & a_2 & \cdots &0\\
#                    \vdots & \vdots & \ddots & \vdots\\
#                    0 &\cdots &0&a_n
# 		\end{pmatrix}
# $$
# 
# If we take the second power of this matrix, i.e. $A^2 = AA$, the resulting matrix is
# 
# $$
# A^2 = AA = \begin{pmatrix}a_1 & 0 &\cdots &0\\
# 									 0   & a_2 & \cdots &0\\
#                    \vdots & \vdots & \ddots & \vdots\\
#                    0 &\cdots &0&a_n
# 		\end{pmatrix}\begin{pmatrix}a_1 & 0 &\cdots &0\\
# 									 0   & a_2 & \cdots &0\\
#                    \vdots & \vdots & \ddots & \vdots\\
#                    0 &\cdots &0&a_n
# 		\end{pmatrix} = \begin{pmatrix}a_1^2 & 0 &\cdots &0\\
# 									 0   & a_2^2 & \cdots &0\\
#                    \vdots & \vdots & \ddots & \vdots\\
#                    0 &\cdots &0&a_n^2
# 		\end{pmatrix}
# $$
# 
# Hence we see that the second power of a diagonal matrix is obtained by simply squaring each entry. 
# More generally, the $k^{th}$ power of a diagonal matrix can be seen to be
# 
# 
# $$
# A^k = \begin{pmatrix}a_1^k & 0 &\cdots &0\\
# 									 0   & a_2^k & \cdots &0\\
#                    \vdots & \vdots & \ddots & \vdots\\
#                    0 &\cdots &0&a_n^k
# 		\end{pmatrix}
# $$
# 
# 
# Let's verify this in Python. 
# First, we define a diagonal matrix $A$ using the `np.diag` function.

# In[1]:


import numpy as np

A = np.diag([1,2,3,4,5,6,7,8,9,10])
A


# Before practicing taking powers of this matrix, let's take a moment to review how we can take powers in Python more generally. 
# We can find the $k^{th}$ power of a scalar $x$ in Python by using the `**` operation. For example,

# In[2]:


x = 2
print('x^2 = ', x**2)
print('x^3 = ', x**3)
print('x^4 = ', x**4)


# The same operation can be used to take the _element-wise_ power of a numpy array. 
# For example,

# In[3]:


x = np.array([1,2,3,4,5])
x**2


# This returns the array `[1,4,9,16,25]`, which are the squares of each of the entries of `[1,2,3,4,5]`. 
# The same applies to 2-d arrays, i.e., matrices.
# In this case, we get the _element-wise_ power of the matrix.

# In[4]:


A**2


# As we saw above, for diagonal matrices, this is the same as the second power of the matrix. 
# Let's verify this.

# In[5]:


AA = np.dot(A,A)
np.allclose(AA, A**2)


# This is a very special property of diagonal matrices.
# In general, the $k^{th}$ power of a matrix $A$ will not be the same as taking the $k^{th}$ power of each entry of $A$, as we will see later on.
# 
# 
# ### Powers of a diagonal matrix as $k\to \infty$
# 
# A natural question to ask when taking power of a matrix is what happens to the matrix $A^k$ when we let $k$ get large. 
# Does the matrix $A^k$ converge to a finite matrix $A_\infty$? 
# Or do the entries diverge? 
# 
# In the case of a diagonal matrix $A$, this question is easy to answer, since we know that $A^k$ corresponds to taking the $k^{th}$ power of each of the diagonal entries of $A$. 
# Thus, our question can be answered by asking what happens to the scalar $x^k$ as $k\to\infty$. 
# Indeed, we know the following are true:
# 
# 
# $$
# \lim_{k\to\infty}x^k = \begin{cases} \infty & \text{if } x>1\\ 1 & \text{if } x=1\\ 0 & \text{if } -1 < x < 1\\ \text{does not exist} & \text{if } x\leq -1\end{cases}
# $$
# 
# 
# Hence, we can expect to have a finite limit $A^k$ as $k\to \infty$ if the diagonal entries of $a_1,\dots,a_n$ of $A$ satisfy $|a_i|<1$ or $a_i = 1$ for all $i=1,\dots,n$. 
# Otherwise, the matrix $A^k$ will either oscillate or diverge as $k\to\infty$. 
# Let's see an example with our matrix $A$ from above. 
# 
# To get an idea of what $A^k$ is as $k\to\infty$, let's try computing $A^{10}$.

# In[6]:


k = 10

Ak = np.linalg.matrix_power(A, k)
Ak


# As would expect based on our discussion above, the entries are growing very quickly. 
# On the other hand, let's see what happens when we _rescale_ the matrix $A$ between each successive multiplication. 
# In this case, a natural quantity to rescale a matrix $A$ by is the reciprocal of its largest diagonal entry, which in this case is $\frac{1}{10}$.

# In[7]:


A_rescaled = (1./10)*A
A_rescaled_k = np.linalg.matrix_power(A_rescaled, k)
np.round(A_rescaled_k,4)


# Here, we rounded the matrix to $4$ decimals to make it more readable. 
# We see that the entries aren't growing, and are actually getting smaller, with the exception of $10$th diagonal which stays constant at $1$. 
# Let's try what happens with $k=100$ now to get a better idea for what the limiting matrix is.

# In[8]:


k = 100

A_rescaled_k = np.linalg.matrix_power(A_rescaled, k)
np.round(A_rescaled_k,4)


# Here it is easy to see that the matrix $A^k$ converges to the matrix of all zeros except the last diagonal entry, which is still $1$. 
# 
# This probably seems like a special situation, but it is very common---if you view the matrix the right way.
# 
# 
# ## Powers of a genenal (symmetric) matrix
# 
# Let's now consider how taking powers of matrices is similar/different when considering powers of a general (symmetric) matrix.
# 
# Let's define a symmetric matrix $A$ to use in this section. 
# Here we use a trick to do this: for any matrix $B$, the matrix $B+B^\top$ is always symmetric.
# (If we wanted, we could take advantage of the fact that for any matrix $B$, the matrix $BB^\top$ is always symmetric; or we could use any symmetric matrix.)

# In[9]:


B = np.arange(1, 26).reshape(5,5)
A = .5*(B + B.T)
A


# The first fact about general symmetric matrices which we verify here is that for non-symmetric $A$, it is emphatically _not_ the case that $A^k$ is the same as taking the $k^{th}$ power of each of the entries of $A$. 
# Let's check this for our case.
# 
# The entry-wise $k^{th}$ power of $A$ is

# In[10]:


k = 2

A**k


# While the $A^k$ is given by

# In[11]:


np.linalg.matrix_power(A, k)


# Indeed, these two matrices are very much not the same. 
# If we think about what matrix multiplication does, this of course makes sense: the $(i,j)$th entry of the product $AA$ depends on the entire $i^{th}$ row and $j^{th}$ column of $A$, whereas the $(i,j)^{th}$ entry of the entry-wise square of $A$ only requires calculating $a_{ij}^2$. 
# 
# 
# ### Powers of a genenal (symmetric) matrix as $k\to \infty$
# 
# For diagonal matrices, we saw that the power $A^k$ converges to a finite matrix only if we rescale things correctly. 
# Here, we will see that this is also true for general symmetric matrices, but that we need to be a bit careful about how we rescale them.
# 
# First, let's verify that without rescaling, the matrix $A^k$ will diverge as $k\to\infty$.

# In[12]:


k = 10

Ak = np.linalg.matrix_power(A, k)
Ak


# Here, $A^k$ diverges due to the specific matrix $A$.  For other matrices, $A^k$ would converge to the all-zeros matrix.
# 
# Indeed, even for $k=10$, the matrix entries blow up. 
# So how should we rescale $A$ so that this doesn't happen? 
# We might be inspired by the diagonal case and rescale the matrix by the reciprocal of the largest entry of $A$, which in this case would be $\frac{1}{25}$. 
# Let's see if this works by looking at $(\frac{1}{25}A)^k$ for $k=20$.

# In[13]:


k = 20

A_rescaled1 = (1./25)*A
A_rescaled1_k = np.linalg.matrix_power(A_rescaled1, k)
A_rescaled1_k


# Unfortunately, this idea doesn't work---the entries still seem to blow up. 
# However, let's try rescaling by the constant $\frac{1}{\lambda}$ where $\lambda = 71.3104367$. 
# (We'll see later why this particular rescaling makes sense.)

# In[14]:


k = 20
lam = 71.3104367

A_rescaled2 = (1./lam)*A
A_rescaled2_k = np.linalg.matrix_power(A_rescaled2, k)
A_rescaled2_k


# Things don't seem to be blowing up this time, so let's see what happens with $k=100$.

# In[15]:


k = 100
A_rescaled2_k = np.linalg.matrix_power(A_rescaled2, k)
A_rescaled2_k


# Indeed, taking higher powers doesn't seem to be changing the matrix anymore, so this rescaling works to guarantee convergence of the matrix powers. 
# So, how did we come up with the mysterious constant $\lambda = 71.3104367$? 
# This turns out to be a special number which is the largest _eigenvalue_ of $A$. 
# We won't be discussing the topic of eigenvalues until a bit later in the semester, but consider this some interesting motivation for the topics that we'll study later on. 
# When we get to this topic later on, it will be more clear where this constant came from, and why rescaling by its reciprocal works.
