#!/usr/bin/env python
# coding: utf-8

# # Changing Basis
# 
# We previously introduced the concept of a _basis_ for a vector space $V$such as $\mathbb{R}^n$. 
# Recall that a set $v_1,\dots,v_k$ of vectors is called a _basis_ for $V$ if 
# 
# - $\text{span}(v_1,\dots,v_k) = V$, and
# - $v_1,\dots,v_k$ are linearly independent.
# 
# If $V = \mathbb{R}^{n}$, then we have $k=n$ vectors in any basis for $\mathbb{R}^{n}$.
# In this case, any set of $n$ linearly independent vectors form a basis.
# 
# In this section, we will demonstrate the following.
# 
# - How vectors are represented with respect to bases, and how a given vector can have different numerical values when represented with respect to different bases.
# - How matrices are represented with respect to bases, and how a given matrix can have different numerical values when represented with respect to different bases.
# - The tranformation that taking in coordinates $(\alpha_1,\dots, \alpha_n)$ of a vector $x$ with respect to a basis $B_1$, and returning the coordinates $(\beta_1,\dots, \beta_n)$ with respect to a different basis $B_2$ is a linear transformation. We will show you how to find the linear map $T : (\alpha_1,\dots, \alpha_n) \mapsto (\beta_1,\dots,\beta_n)$, which is a matrix called the _change of basis matrix_.
# - Linear maps $A$ (or equivalently matrices) are also implicitly represented with respect to a particular basis. We will show you how to represent such linear maps with respect to a different basis. 
# 
# 
# ## Representing vectors with respect to bases
# 
# A natural basis to use in many contexts is the standard (or canonical) basis $E = \{e_1,\dots, e_n \}$, where
# 
# $$
# e_i = \begin{pmatrix}0\\ \vdots \\ 1 \\ \vdots \\0\end{pmatrix}  .
# $$
# 
# That is, $e_i$ is $1$ in the $i$th component, and $0$ elsewhere. 
# 
# The reason that this is such a convenient basis is the following.
# When we write a vector 
# 
# 
# $$
# x = \begin{pmatrix} x_1 \\ \vdots \\ \\ x_n \end{pmatrix} \in \mathbb{R}^n,
# $$
# 
# we are implicitly writing it with respect to the standard basis.
# The reason for this is that 
# $$
# x = x_1 e_1 + x_2e_2 + \dots + x_ne_n  .
# $$
# 
# _Said another way, a vector is an abstract thing that obeys the rules of scalar multiplication and vector addition, but when we represent a vector as an array or list of numbers, we are representing that vector with respect to the standard basis._
# 
# On the other hand, if $B = \{v_1,\dots, v_n\}$ is a different basis for $\mathbb{R}^n$, we could equivalently represent the vector $x$ with respect to this basis as $x = \alpha_1 v_1 + \alpha_2v_2 + \dots + \alpha_nv_n$ for some scalars $\alpha_1,\dots,\alpha_n$. Here $(\alpha_1,\dots,\alpha_n)$ are called the coordinates of $x$ with respect to $B$. 
# 
# ## The change of basis matrix
# 
# ### Changing to and from the standard basis
# 
# Let's start by assuming we have a vector $x = (x_1,\dots, x_n)$ represented with respect to the standard basis. We want to find the coordinates $\alpha = (\alpha_1,\dots,\alpha_n)$ of this vector with respect to a new basis $B = \{v_1,\dots,v_n\}$. That is, we want to find numbers $\alpha_1,\dots,\alpha_n$ satisfying the following:
# 
# 
# $$
# x = \alpha_1 v_1 + \alpha_2v_2 + \dots + \alpha_nv_n
# $$
# 
# 
# Let's define the following matrix:
# 
# 
# $$
# V = \begin{pmatrix}v_1 & v_2 & \cdots & v_n\end{pmatrix}
# $$
# 
# 
# i.e. the $n\times n$ matrix whose columns are the vectors $v_1,\dots, v_n$. Then if we recall the definition of matrix-vector multiplication, we realize that we can conveniently write 
# 
# 
# $$
# x = \alpha_1 v_1 + \alpha_2v_2 + \dots + \alpha_nv_n = V\alpha
# $$
# 
# 
# The equation $x = V\alpha$ is important: in fact, it immediately gives us the equation to change from the basis $B$ to the standard basis $E$: given a representation $\alpha$ of $x$ with respect to $B$, we can simply apply the linear transformation $V$ to $\alpha$ and get back the coordinates of $x$ with respect to the standard basis. 
# 
# But we would like to go the other way: change from the standard basis $E$ to the new basis $B$. To do this, all we need to do is _invert_ the transformation $V$. Indeed, it is easy to show that $V$ is invertible, since it is square and has linearly independent columns. Therefore we have
# 
# 
# $$
# x = V\alpha \iff \alpha = V^{-1}x
# $$
# 
# 
# Therefore, we've seen that we can take a vector $x$ represented with respect to the standard basis $E$ and obtain the coordinates of $x$ with respect to a different basis $B$ by applying the linear transformation $V^{-1}$. For the remainder of this section, we denote $T_{E\to B} = V^{-1}$ as the _change of basis matrix_ from $E$ to $B$. Notice that we also have $T_{B\to E} = V$, and that we always have the relation
# 
# 
# $$
# T_{E\to B} = T_{B\to E}^{-1}
# $$
# 
# 
# Let's see a few examples of changing bases in Python. Let's start out with a vector $x = (1,2,3,4)$ represented with respect to the standard basis

# In[1]:


import numpy as np

x = np.array([1,2,3,4])
x


# Suppose we want to represent this with respect to the following basis:
# 
# 
# $$
# v_1 = \begin{pmatrix}1\\ 0\\0\\0\end{pmatrix},\;\;\; v_2 = \begin{pmatrix}1\\ 1\\0\\0\end{pmatrix},\;\;\; v_3 = \begin{pmatrix}1\\ 1\\1\\0\end{pmatrix},\;\;\; v_4 = \begin{pmatrix}1\\ 1\\1\\1\end{pmatrix}
# $$

# In[2]:


v1 = np.array([1,0,0,0])
v2 = np.array([1,1,0,0])
v3 = np.array([1,1,1,0])
v4 = np.array([1,1,1,1])


# As we saw above, we need to form the matrix $V$ whose columns are $v_1,v_2,v_3,v_4$. We can do this in numpy with the following.

# In[3]:


V = np.stack([v1,v2,v3,v4], axis=1)
V


# Now we want to find the change of basis matrix $T_{E\to B} = V^{-1}$. To do this, we need to _invert_ the matrix $V$. We can do this in numpy with the function `np.linalg.inv()`.

# In[4]:


T_EtoB = np.linalg.inv(V) 
T_EtoB


#  To compute the coordinates $\alpha$ of $x$ with respect to $v_1,\dots,v_4$, we just need to apply this tranformation to $x$:

# In[5]:


alpha = np.dot(T_EtoB, x)
alpha


# Now let's check that this actually gave us the correct answer by computing $\alpha_1v_1 + \alpha_2v_2 + \alpha_3v_3 + \alpha_4v_4$:

# In[6]:


alpha[0]*v1 + alpha[1]*v2 + alpha[2]*v3 + alpha[3]*v4


# Indeed, we obtain the original vector $x$ back, and so $\alpha$ are the correct coordinates for $x$ with respect to $v_1,\dots,v_4$.
# 
# We can also get the vector $x$ back by applying the map $T_{B\to E} = V$:

# In[7]:


T_BtoE = V
np.dot(T_BtoE, alpha)


# As expected, this also gives us the original vector $x$ back.
# 
# ### Changing to and from two arbitrary bases
# 
# In the previous section, we saw how to derive the change of basis matrix between the standard basis and another non-standard matrix. Here, we see how to use this concept to find the change of basis matrix between two arbitrary bases. Suppose we have two bases $B_1 =\{ v_1,\dots,v_n\}$ and $B_2 = \{u_1,\dots,u_n\}$ for $\mathbb{R}^n$. Like before, let's define $V = \begin{pmatrix} v_1,\dots,v_n\end{pmatrix}$,  $U = \begin{pmatrix} u_1,\dots,u_n\end{pmatrix}$ be the matrices whose columns are the basis vectors of $B_1$ and $B_2$, respectively.
# 
# In the previous section, we saw that the change of basis matrix for changing from the standard basis $E$ to $B_1$ or $B_2$, and back again. In particular, we have that
# 
# 
# $$
# T_{E\to B_1} = V^{-1},\;\;\; T_{B_1\to E} = V,\;\;\; T_{E\to B_2} = U^{-1},\;\;\; T_{B_2\to E} = U 
# $$
# 
# 
# Now we can derive the change of basis matrix from $B_1$ to $B_2$ and vice versa by _composing_ these changing of basis matrices. Our strategy is as follows: say we start with coordinates $\alpha = (\alpha_1,\dots,\alpha_n)$ in terms of $B_1$. We first apply $T_{B_1\to E}$, to obtain the coordinates in terms of the standard basis, and then apply $T_{E\to B_2}$ to obtain coordinates $\beta = (\beta_1,\dots,\beta_n)$ in terms of the basis $B_2$. This gives us
# 
# 
# $$
# \beta = T_{E\to B_2}T_{B_1\to E}\alpha = U^{-1}V\alpha
# $$
# 
# 
# Hence we see that the change of basis matrix from $B_1\to B_2$ is simply
# 
# 
# $$
# T_{B_1 \to B_2} = U^{-1}V.
# $$
# 
# 
# Similarly, the change of basis matrix to go back, i.e. from $B_2\to B_1$ is simply
# 
# 
# $$
# T_{B_2\to B_1} = T_{E\to B_1}T_{B_2 \to E} = V^{-1}U.
# $$
# 
# 
# And that's all there is to it! By _composing_ linear maps, we can obtain the linear map changing between any two arbitrary bases.
# 
# Let's see some examples in Python.
# 
# We'll use the same basis $B_1 =\{ v_1,v_2,v_3,v_4\}$ defined above, as well as the basis $B_2 = \{u_1, u_2,u_3,u_4\}$ defined below:
# 
# 
# $$
# u_1 = \begin{pmatrix}-1\\ 0\\0\\0\end{pmatrix},\;\;\; u_2 = \begin{pmatrix}1\\ -1\\0\\0\end{pmatrix},\;\;\; u_3 = \begin{pmatrix}1\\ 1\\-1\\0\end{pmatrix},\;\;\; u_4 = \begin{pmatrix}-1\\ 1\\1\\-1\end{pmatrix}
# $$
# 
# 
# We define these as numpy arrays:

# In[8]:


u1 = np.array([-1, 0, 0, 0])
u2 = np.array([1, -1, 0, 0])
u3 = np.array([1, 1, -1, 0])
u4 = np.array([-1, 1, 1, -1])


# Let's store these in a matrix $U$.

# In[9]:


U = np.stack([u1,u2,u3,u4], axis=1)
U


# Now we can compute the tranformations $T_{B_1 \to B_2}$ and $T_{B_2 \to B_1}$ using the formulas above.

# In[10]:


T_B1toB2 = np.dot(np.linalg.inv(U), V)
T_B2toB1 = np.dot(np.linalg.inv(V), U)


# Let's see what the coordinates of $\alpha$ are with respect to the basis $B_2$:

# In[11]:


beta = np.dot(T_B1toB2, alpha)
beta


# We should be able to confirm that $x = \beta_1 u_1 + \beta_2 u_2 + \beta_3 u_3 + \beta_4 u_4$:

# In[12]:


beta[0]*u1 + beta[1]*u2 + beta[2]*u3 + beta[3]*u4


# As expected, this gives us our vector $x$, now represented with respect to the basis $B_2$. We should also we able to confirm that we get $\alpha$ back when we apply the transformation $T_{B_2 \to B_1}$ to $\beta$:

# In[13]:


np.dot(T_B2toB1, beta)


# Indeed, we get back the coordinates $\alpha$ with respect to $B_1$.
# 
# ## Representing a matrix with respect to a basis
# 
# Suppose we have a linear function $f(x)$, which is represented as $f(x) = Ax$ for a vector $x$ represented with respect to the standard basis. Then $(i,j)$th entry of $A$ is the $i$th coordinate of the vector $f(e_j)$, where $e_j$ is the $j$th standard basis vector. To see this, notice that
# 
# 
# $$
# f(e_j) = Ae_j = \begin{pmatrix}a_{11} & \cdots & a_{1n}\\ \vdots &\ddots &\vdots\\ a_{n1} & \cdots & a_{nn}\end{pmatrix}\begin{pmatrix}0\\ \vdots \\ 1 \\ \vdots \\0\end{pmatrix} = \begin{pmatrix}a_{1j}\\ a_{2j}\\\vdots \\ a_{nj}\end{pmatrix}
# $$
# Thus we see that the $i$th entry of the vector $Ae_j$ is $a_{ij}$. Hence if we want to represent the vector $f(e_j)$ with respect to the standard basis, we can do so with the representation 
# 
# 
# $$
# f(e_j) = a_{1j}e_1 + a_{2j}e_2 + \cdots + a_{nj}e_j
# $$
# More generally, this also works with other bases. For example, consider a basis $B = \{v_1,\dots, v_n\}$. If $A_B$ is a matrix representing $f$ with respect to this basis, then the $(i,j)$th entry of $A_B$ is the $i$th coordinate of the vector $f(v_j)$ with respect to $B$. In what follows, we show how we can find the matrix $A_B$ given the linear transformation $f(x) = Ax$ represented with respect to the standard basis.
# 
# ### Changing basis for a matrix
# 
# Suppose we have a vector $x\in \mathbb{R}^n$, represented with respect to the standard basis $E$ as $x = (x_1,\dots,x_n)$. We know that we can transform such a vector with a matrix $A \in \mathbb{R}^{m\times n}$, representing a linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$. Now suppose we chose to change basis, and represent $x$ with respect to a new basis $B = v_1,\dots, v_n$ using the change of basis matrix $T_{E\to B}$ to get the new coordinates $\alpha = T_{E\to B}x$. If we directly apply the matrix $A$ to the new coordinates $\alpha$, we won't in general get the same result as if we applied it to the original vector $x$. This is because when we represent a linear map as a matrix, we are also implicitly doing so with a fixed basis. However, we will see in this section that we can also represent a linear map (i.e. a matrix) with respect to different bases.
# 
# To see how this works, we will use a similar approach as in the previous section. Since $A$ is represented with respect to the standard basis, we need to pass it vectors represented with respect to this same basis. Therefore, a natural approach when trying to apply the linear map $A$ to vectors $\alpha$ represented with respect to $B$ is to first transform $\alpha$ to be in terms of the standard basis, using $T_{B\to E}$, then apply the linear map $A$, and then finally tranform back to the basis $B$. In symbols, we apply the following linear map:
# 
# 
# $$
# A_B = T_{E\to B}AT_{B\to E}
# $$
# 
# 
# Here we use the notation $A_B$ to denote the linear map $A$ represented with respect to the basis $B$. If we let $V = \begin{pmatrix} v_1 &\cdots & v_n\end{pmatrix}$ be the matrix whose columns are the basis vectors of $B$, we can alternatively write the matrix $A_B$ as 
# 
# 
# $$
# A_B = V^{-1}AV
# $$
# 
# 
# We will see formulas of this kind frequently later in the course when we discuss eigenvalue and singular value decompositions, and so it is important to understand that these decompositions are really just representing a particular linear transformation with respect to a new basis.
# 
# Let's see some example of how this works in Python, using the same basis $v_1,v_2,v_3,v_4$ defined above. Let's consider the linear map represented with respect to the standard basis as 
# 
# 
# $$
# A = \begin{pmatrix}1 & 1 & 1 & -1\\ 1 & 1 & -1 & 1\\ 1& -1 & 1 &1\\ -1 & 1 & 1& 1\end{pmatrix}
# $$
# 
# 
# In numpy, we can define this with the following code.

# In[14]:


A = np.array([[1, 1, 1, -1], [1,1,-1,1], [1, -1, 1, 1], [-1,1,1,1]])
A


# Now let's see what answer we get when we apply this matrix to the vector $x = (1,2,3,4)$.

# In[15]:


Ax = np.dot(A,x)
Ax


# This gives us a new vector $Ax$ represented with repsect to the standard basis $E$. 
# 
# Now suppose that instead we choose to work with the basis $B$, and use the representation of $x$ in terms of the coordinates $\alpha$ in this basis. As we saw above, we can compute the matrix $A$ with respect to the basis $B$ as $A_B = T_{E\to B}AT_{B\to E}$. We do this in Python below.

# In[16]:


A_B = np.dot(T_EtoB, np.dot(A, T_BtoE))
A_B


# Now let's apply this to the coordinates $\alpha$ to get $A_B\alpha$.

# In[17]:


A_Balpha = np.dot(A_B, alpha)
A_Balpha


#  If we did everything correctly, we should see that this gives us the coordinates of the vector $Ax$ with respect to the basis $B$. Let's check that this is true by computing $[A_B\alpha]_1v_1 + [A_B\alpha]_2v_2 +[A_B\alpha]_3v_3 +[A_B\alpha]_4v_4 $.

# In[18]:


A_Balpha[0]*v1 + A_Balpha[1]*v2 + A_Balpha[2]*v3 + A_Balpha[3]*v4


# As expected, this indeed gives us the same thing as $Ax$!
