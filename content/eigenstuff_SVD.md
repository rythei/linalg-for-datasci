---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# The Singular Value Decomposition

In the previous workbook, we saw that symmetric square matrices $\boldsymbol{A}$ have a special decomposition called an _eigenvalue_ decomposition: $\boldsymbol{A} = \boldsymbol{V\Lambda V}^\top$, where $\boldsymbol{V}$ is an orthogonal matrix satisfying $\boldsymbol{V^\top V} = \boldsymbol{VV}^\top = \boldsymbol{I}$, whose columns are _eigenvectors_, and $\boldsymbol{\Lambda} = \text{diag}(\lambda_1,\dots, \lambda_n)$ is a diagonal matrix containing the _eigenvalues_ of $\boldsymbol{A}$.

In this section, we see that _all_ matrices -- even non-square and non-symmetric -- have a similar decomposition called the _singular value decomposition_, or SVD. Let's first remind ourselves why we can't eigenvalue decomposition doesn't make sense for non-square matrices. Suppose $\boldsymbol{A}$ is a $m\times n$ matrix. Then for any $\boldsymbol{v}\in \mathbb{R}^n$, $\boldsymbol{Av}$ is a vector in $\mathbb{R}^m$, and so the eigenvalue condition $\boldsymbol{Av} = \lambda \boldsymbol{v}$ does not make sense in this setting: the left-hand side is a $m$-dimensional vector, while $\lambda \boldsymbol{v}$ is an $n$-dimensional vector.

Instead, for $m\times n$ matrices $\boldsymbol{A}$, we consider instead a generalized version of the eigenvalue condition: vectors $\boldsymbol{v}\in \mathbb{R}^n$, $\boldsymbol{u}\in \mathbb{R}^m$ and a number $\sigma$ are called _right and left singular vectors_, and a _singular value_ if they satisfy:


$$
\begin{aligned}
\boldsymbol{Av} = \sigma \boldsymbol{u} && (1)\\
\boldsymbol{A^\top u} = \sigma \boldsymbol{v} && (2)
\end{aligned}
$$


Singular values and singular vectors are in fact closely related to eigenvalues and eigenvectors. Let's see why this is the case. Let's start with equation $(1)$, and multiply both sides by $\boldsymbol{A}^\top$:


$$
\boldsymbol{Av} = \sigma \boldsymbol{u} \implies \boldsymbol{A^\top A v} = \sigma \boldsymbol{A^\top u}.
$$


Now, let's plug in equation $(2)$, which says that $\boldsymbol{A^\top u} = \sigma \boldsymbol{v}$. We get:


$$
\boldsymbol{A^\top A v} = \sigma \boldsymbol{A^\top u} = \sigma^2 \boldsymbol{v}.
$$


This looks more like something we've seen before: if we set $\boldsymbol{B} = \boldsymbol{A^\top A}$ and $\lambda = \sigma^2$, this can be written as $\boldsymbol{Bv} = \lambda \boldsymbol{v}$. Therefore, the squared singular values and right singular vectors can be obtained by computing an eigenvalue decompostion of the symmetric matrix $\boldsymbol{A^\top A}$. Using a similar derivation, we can also show that


$$
\boldsymbol{AA^\top u} = \sigma^2 \boldsymbol{u}
$$


from which we see that $u$ is really an eigenvector of the symmetric matrix $AA^\top$.

**Remark:** In our discussion above, we saw that the eigenvalues of $\boldsymbol{AA}^\top$ and/or $\boldsymbol{A^\top A}$ correspond to the _squared_ singular values $\sigma^2$ of $\boldsymbol{A}$. This may seem odd, since we know that in general matrices may have positive or negative eigenvalues. However, this occurs specifically because $\boldsymbol{A^\top A}$ and $\boldsymbol{AA}^\top$ are always _positive semi-definite_, and therefore always have non-negative eigenvalues. To see why this is true, note that the smallest eigenvalue of $\boldsymbol{A^\top A}$ are the minimum of the quadratic form $Q(\boldsymbol{x}) = \boldsymbol{x^\top A^\top A x}$, over all unit vectors $\boldsymbol{x}$. Then:


$$
\lambda_{\text{min}} = \min_{\|\boldsymbol{x}\|_2 =1} \boldsymbol{x^\top A^\top A x} = \min_{\|\boldsymbol{x}\|_2 =1} (\boldsymbol{Ax})^\top \boldsymbol{Ax} = \min_{\|\boldsymbol{x}\|_2 =1}\|\boldsymbol{Ax}\|_2^2 \geq 0.
$$


A similar derivation shows that all the eigenvalues of $\boldsymbol{AA}^\top$ are non-negative.

How many singular values/vectors do we expect to get for a given $m\times n$ matrix $\boldsymbol{A}$? We know that the matrix $\boldsymbol{A^\top A}$ is $n\times n$, which gives us $n$ eigenvectors $\boldsymbol{v}_1,\dots, \boldsymbol{v}_n$ (corresponding to $n$ right singular vectors of $\boldsymbol{A}$), and $\boldsymbol{AA}^\top$ is $m\times m$, giving us $m$ eigenvectors $\boldsymbol{u}_1,\dots, \boldsymbol{u}_m$ (corresponding to $m$ left singular vectors of $\boldsymbol{A}$). The matrices $\boldsymbol{A^\top A}$ and $\boldsymbol{AA}^\top$ will of course not have the same number of eigenvalues, though they do always have the same _non-zero_ eigenvalues. The number $r$ of nonzero eigenvalues of $\boldsymbol{A^\top A}$ and/or $\boldsymbol{AA}^\top$  is exactly equal to the _rank_ of $\boldsymbol{A}$, and we always have that $r \leq \min(m,n)$.

Now let's collect the vectors $\boldsymbol{u}_1,\dots, \boldsymbol{u}_m$ into an $m\times m$ matrix $\boldsymbol{U} = \begin{bmatrix} \boldsymbol{u}_1 & \cdots & \boldsymbol{u}_m\end{bmatrix}$ and likewise with $\boldsymbol{v}_1,\dots, \boldsymbol{v}_n$ into the $n\times n$ matrix $\boldsymbol{V} = \begin{bmatrix}\boldsymbol{v}_1 &\cdots & \boldsymbol{v}_n\end{bmatrix}$. Note that since $\boldsymbol{U}$ and $\boldsymbol{V}$ come from the eigenvalue decompositions of the symmetric matrices $\boldsymbol{AA}^\top$ and $\boldsymbol{A^\top A}$, we have that $\boldsymbol{U}$ and $\boldsymbol{V}$ are always orthogonal, satisfying $\boldsymbol{U^\top U} = \boldsymbol{UU}^\top = \boldsymbol{I}$ and $\boldsymbol{V^\top V} = \boldsymbol{VV}^\top = \boldsymbol{I}$.

Then let's define the $m\times n$ matrix $\boldsymbol{\Sigma}$ as follows:


$$
\boldsymbol{\Sigma}_{ij} = \begin{cases}\sigma_i & \text{if } i=j\\ 0  & \text{if } i\neq j\end{cases}
$$


That is, $\boldsymbol{\Sigma}$ is a "rectangular diagonal" matrix, whose diagonal entries are the singular values of $\boldsymbol{A}$ -- i.e. the square roots of the eigenvalues of $\boldsymbol{A^\top A}$ or $\boldsymbol{AA}^\top$. For example, in the $2\times 3$ case $\boldsymbol{\Sigma}$ would generically look like


$$
\begin{bmatrix}\sigma_1 & 0 & 0 \\ 0 & \sigma_2 &0\end{bmatrix}
$$

and in the $3\times 2$ case it would look like


$$
\begin{bmatrix}\sigma_1 & 0  \\ 0 & \sigma_2 \\ 0 & 0\end{bmatrix}.
$$



Given the matrices $\boldsymbol{U}, \boldsymbol{\Sigma}$ and $\boldsymbol{V}$, we can finally write the full singular value decomposition of $A$:

$$
\boldsymbol{A} = \boldsymbol{U\Sigma V}^\top.
$$


This is one of the most important decompositions in linear algebra, especially as it relates to statistics, machine learning and data science.

**Remark:** Sometimes you may see a slightly different form of the SVD: the rank of $\boldsymbol{A}$ is $r\leq \min(n,m)$, we can actually remove the last $m-r$ columns of $\boldsymbol{U}$ and $n-r$ column of $\boldsymbol{V}$ (so that $\boldsymbol{U}$ is $m\times r$ and $\boldsymbol{V}$ is $n\times r$), and let $\boldsymbol{\Sigma}$ be the $r\times r$ diagonal matrix $\text{diag}(\sigma_1,\dots,\sigma_r)$. The two forms are totally equivalent, since the last $m-r$ columns of $\boldsymbol{U}$ are only multiplied by the $m-r$ zero rows at the bottom of $\boldsymbol{\Sigma}$ anyway. This form is sometimes called the "compact SVD". In this workbook, we'll assume we're working with the "standard" version, introduced above, though the compact version is sometimes better to work with in practice, especially when the matrix $\boldsymbol{A}$ is very low rank, with $r\ll m,n$.

## Computing the SVD in Python

Let's see some examples of computing the singular value decomposition in Python.

First, let's draw a random $m\times n$ matrix $\boldsymbol{A}$ to use.

```{code-cell}
import numpy as np
np.random.seed(1)

m = 5
n = 3

A = np.random.normal(size=(m,n))
```

Next, let's compute the eigenvalue decompositions of $\boldsymbol{A^\top A} = \boldsymbol{V\Lambda}_1 \boldsymbol{V}^\top$ and $\boldsymbol{AA^\top} = \boldsymbol{U\Lambda}_2 \boldsymbol{U}^\top$.

```{code-cell}
AAT = np.dot(A,A.T)
ATA = np.dot(A.T,A)

Lambda1, V = np.linalg.eig(ATA)
Lambda2, U = np.linalg.eig(AAT)
```

Of course, since $\boldsymbol{A^\top A}$ and $\boldsymbol{AA}^\top$ are of different dimensions, $\boldsymbol{\Lambda}_1$ and $\boldsymbol{\Lambda}_2$ will also be of different dimensions. However, as we mentioned above, $\boldsymbol{\Lambda}_1$ and $\boldsymbol{\Lambda}_2$ should have the same _non-zero_ entries. Let's check that this is true.

```{code-cell}
print(Lambda1.round(8))
print(Lambda2.round(8))
```

Indeed, we get the same non-zero eigenvalues, but $\boldsymbol{\Lambda}_2$ has 10 extra zero eigenvalues. Now let's form the matrix $\boldsymbol{\Sigma}$, which will be $m\times n$ matrix with $\boldsymbol{\Sigma}_{ii} = \sqrt{\lambda_i}$ and $\boldsymbol{\Sigma}_{ij} = 0$ for $i\neq j$.

```{code-cell}
Sigma = np.zeros((m,n))
for i in range(n):
    Sigma[i,i] = np.sqrt(Lambda1[i])

Sigma
```

Now we have our matrices $\boldsymbol{V},\boldsymbol{U}$ and $\boldsymbol{\Sigma}$; let's check that $\boldsymbol{A} = \boldsymbol{U\Sigma V}^\top$.

```{code-cell}
np.allclose(A, np.dot(U, np.dot(Sigma, V.T)))
```

Strangely, this doesn't give us the correct answer. The reason is that we have an issue with one of the signs of the eigenvectors: the eigenvalue is invariant to switching the signs of one of the eigenvectors (i.e. multiplying one of the columns of $V$ or $U$ by $-1$ ), but the SVD is not. Since we computed the eigenvalue decomposition of $\boldsymbol{A^\top A}$ and $\boldsymbol{AA}^\top$ separately, there was no guarantee that we would get the correct signs of the eigenvectors. It turns out in this case we can fix this by switching the sign of the third column of $\boldsymbol{V}$.

```{code-cell}
V[:,2] *= -1
np.allclose(A, np.dot(U, np.dot(Sigma, V.T)))
```

Now everything works! However, this issue is a bit annoying in practice -- fortunately, we can avoid it by simply using numpy's build in SVD function, `np.linalg.svd`. Let's see how this works.

```{code-cell}
U, S, VT = np.linalg.svd(A)

Sigma = np.zeros((m,n)) # make diagonal matrix
for i in range(n):
    Sigma[i,i] = S[i]

np.allclose(A, np.dot(U, np.dot(Sigma, VT)))
```

Now that we've seen how the singular value decomposition works in Python, let's explore some interesting applications.

## Approximating matrices with the SVD

While the singular value decomposition appears frequently in statistics and machine learning, one of the most important uses of the SVD is in _low-rank approximation_.

Before explaining the problem of low-rank approximation, let's first state a few useful facts. Let's assume that $\boldsymbol{A}$ is an $m\times n$ matrix with rank $r$ (i.e. $r$ non-zero singular values) and that $\boldsymbol{A} = \boldsymbol{U\Sigma V}^\top$ is its singular value decomposition. Also, let's label $\sigma_1,\dots, \sigma_r$ as its (non-zero) singular values, and let $\boldsymbol{u}_1,\dots,\boldsymbol{u}_r$ be the first $r$ columns of $\boldsymbol{U}$ and $\boldsymbol{v}_1,\dots, \boldsymbol{v}_r$ be the first $r$ columns of $\boldsymbol{V}$. Throughout, we will assume that the singular values are ordered, so that $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r$. Then $\boldsymbol{A}$ can be written as


$$
\boldsymbol{A} = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i\boldsymbol{v}_i^\top.
$$
In the problem of low-rank approximation, we want to find a matrix $\widehat{\boldsymbol{A}}$ which has rank at most $k$, and approximates $\boldsymbol{A}$ closely, i.e. $\boldsymbol{A}\approx \widehat{\boldsymbol{A}}$. Formally, the problem can be written as follows:


$$
\begin{aligned}
\min_{\widehat{\boldsymbol{A}}}&\;\;\;\; \|\boldsymbol{A} - \widehat{\boldsymbol{A}}\|_F && (3)\\
\text{subject to}&\;\;\;\; \text{rank}(\widehat{\boldsymbol{A}}) \leq k
\end{aligned}
$$


Low rank matrices are useful for a variety of reasons, but two important reasons are that 1) they can require less memory to store and 2) we can do faster matrix computations with low rank matrices. It turns out that the solution to the low-rank approximation problem (3) can be exactly constructed from the singular value decomposition. The famous [Eckart–Young–Mirsky theorem](https://en.wikipedia.org/wiki/Low-rank_approximation) states that the solution to the problem (3) is explicitly given by the matrix:


$$
\widehat{\boldsymbol{A}}_k = \sum_{i=1}^k \sigma_i \boldsymbol{u}_i\boldsymbol{v}_i^\top.
$$


That is, the _best possible rank $k$ approximation to $\boldsymbol{A}$ is given by the matrix which keeps just the top $k$ singular values of $\boldsymbol{A}$_. Another way to write $\widehat{\boldsymbol{A}}_k$ is as follows: set $\boldsymbol{\Sigma}_k$ to be the $m\times n$ matrix such that


$$
[\boldsymbol{\Sigma}_k]_{ij} = \begin{cases}\sigma_i & \text{if } i=j \text{ and } i\leq k\\ 0 & \text{otherwise}\end{cases}
$$


That is, $\boldsymbol{\Sigma}_k$ is the same as $\boldsymbol{\Sigma}$, except we set the singular values $\sigma_{k+1},\dots,\sigma_r$ to be equal to zero. Then $\widehat{\boldsymbol{A}}_k = \boldsymbol{U\Sigma}_k\boldsymbol{V}^\top$.

When can we expect $\widehat{\boldsymbol{A}}_k$ to be a good approximation to $\boldsymbol{A}$? It turns out that the error $\|\widehat{\boldsymbol{A}}_k - \boldsymbol{A}\|_F$ is exactly given by $\sqrt{\sum_{i=k+1}^r \sigma_i^2}$. Therefore, if the matrix $\boldsymbol{A}$ has many small singular values, then it can be well approximated by $\widehat{\boldsymbol{A}}_k$.

In the next section, we see an example of low rank approximation with compressing an image.

### An example with image compression

In this section, we look at a simple example of low rank approximation with image compression. Let's see an example image.

```{code-cell}
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np

# load in an example image to use
image = np.load('data/sample_image_1.npy')

# Display image
fig = plt.figure(figsize=(15, 11))
plt.imshow(image, cmap = 'gray')
plt.axis('off')
plt.show()

# Print shape
print('Dimensions:', image.shape)
```

As we can see, this image is of dimension $(427, 640, 3)$, but let's simplify the problem a bit and make it grayscaled. We can do this by taking the average over the third axis.

```{code-cell}
image = image.mean(axis=2)

# Display image
fig = plt.figure(figsize=(15, 11))
plt.imshow(image, cmap = 'gray')
plt.axis('off')
plt.show()

# Print shape
print('Dimensions:', image.shape)
```

Now we can think of this image as a $427 \times 640$ matrix $\boldsymbol{A}$.

Let's compute the SVD of the image.

```{code-cell}
U, s, Vt = np.linalg.svd(image, False)
```

We can check the rank of $\boldsymbol{A}$ by checking how many non-zero singular values it has.

```{code-cell}
print('rank(A) = %s' % len(s[s>0]))
```

So $\boldsymbol{A}$ is a rank $427$ matrix.

In the following we construct the outerproducts of the first eight singular vectors and values, i.e. the first eight terms $\sigma_i \boldsymbol{u}_i\boldsymbol{v}_i^\top$.

```{code-cell}
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15,5))
axs = axs.flatten()
for i in range(8):
    axs[i].imshow(np.outer(U[:,i],Vt.T[:,i])*s[i], cmap = 'gray')
```

These don't appear to look like much, but when we _sum_ them, we can obtain approximations to the original image at various levels of quality.

```{code-cell}
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(15,5))
axs = axs.flatten()
idx = 0
for i in [1,10,20,40,60,100,200, 300]:
    outer_products = [np.outer(U[:,j],Vt.T[:,j])*s[j] for j in range(i)]
    reconstruction = np.sum(np.asarray(outer_products), axis=0)
    axs[idx].imshow(reconstruction, cmap = 'gray')
    axs[idx].set_title('rank k= %s' % i)
    idx += 1

fig.tight_layout()
```

As we can see, as soon as we get to rank $k=40$, we get a fairly good approximation to the original image, and by rank $k=200$ the approximation and the original image are nearly indistinguishable visually.

As we mentioned above, the error from the rank $k$ approximation is given by $\|\widehat{\boldsymbol{A}}_k-\boldsymbol{A}\|_F = \sqrt{\sum_{i=k+1}^{r}\sigma_i^2}$.  Let's plot this error as a function of $k$.

```{code-cell}
sr = np.flip(s**2)
errors = np.cumsum(sr)
errors = np.flip(errors)
errors = np.sqrt(errors)

plt.plot(range(1,428), errors)
plt.xlabel('k')
plt.ylabel('Error of rank k approximation')
plt.show()
```

As we can see, the errors start large when $k$ is small, but quickly get smaller as we increase the rank of our approximation.
