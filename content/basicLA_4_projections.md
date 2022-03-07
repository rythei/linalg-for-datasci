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

# Projections

In this section, we study special linear maps called _projections_. Formally, a projection $P(x)$ is any linear map such that $P^2 = P$. In other words, a projection is simply a _idemptotent_ linear map.

## Projections onto a vector

We begin by considering perhaps the simplest possible projection: a projection onto a single vector. Intuitively, this is probably something you've already seen in high school math. The usual diagram given for this concept is below.

<img src="figs/projection_2d.png" style="zoom:50%;" />

In the above figure, we are projecting a vector $a$ onto a vector $b$. The resulting projection is the vector $\text{proj}_b(a) = a_1$, which is always parallel to $b$. The vector $a_2$ is the "residual" of the projection, which is  $a_2 = a - a_1 = a - \text{proj}_b(a)$. Note that visually from the diagram, we have that $a = a_1 + a_2$, which is of course obvious from the definitions of $a_1$ and $a_2$. We will see below that this diagram is in fact representing a special case of projection onto a vector -- namely, it represents an _orthogonal_ projection.

### Orthogonal projections onto a vector

There is a simple formula for the orthogonal projection of a vector $a \in\mathbb{R}^n$ onto another vector $b\in \mathbb{R}^n$. It is given by the following:



$$
\text{proj}_b(a) = \frac{b^\top a}{b^\top b}b
$$



Notice that $\frac{b^\top a}{b^\top b}$ is just a scalar (assuming $b\neq 0$), and so $\text{proj}_b(a)$ is really just a rescaled version of the vector $b$. This means that for any vector $a$, $\text{proj}_b(a)$ is always parallel to $b$ -- this is why we say that it is a projection "onto" $b$. Why is this called an 'orthogonal' projection? This is because $\text{proj}_b(a)$ is always orthogonal to the "residual" $a - \text{proj}_b(a)$. Let's check that this is in fact true by computing $\text{proj}_b(a)^\top (a - \text{proj}_b(a))$. 



$$
\text{proj}_b(a)^\top (a - \text{proj}_b(a)) = \left(\frac{b^\top a}{b^\top b}b\right)^\top\left(a - \frac{b^\top a}{b^\top b}b\right) = \frac{b^\top a}{b^\top b}\left(b^\top a - b^\top a\frac{b^\top b}{b^\top b}\right) = \frac{b^\top a}{b^\top b}(b^\top a - b^\top a) = 0
$$



Hence the angle between $\text{proj}_b(a)$ and $a - \text{proj}_b(a)$ is always $90^\circ$. You can also see this visually in the figure above.

**Remark:** In the QR decomposition section, we saw the formula $\frac{b^\top a}{b^\top b}b$ appear in the Gram--Schmidt orthogonalization procedure. This is no coincidence: there, we computed



$$
u_j = a_j - \sum_{i = 1}^{j-1}\frac{u_i^\top a_j}{u_i^\top u_i}u_i = a_j - \sum_{i=1}^{j-1}\text{proj}_{u_i}(a_j)
$$



That is, $u_j$ was the residual after projecting $a_j$ onto each of $u_1,\dots, u_{j-1}$. 

As we mentioned previously, the projections we consider are _linear_ maps. We know that all linear maps can be represented as matrices. Let's see how we can represent $\text{proj}_b(a)$ as a matrix transformation. Using the associativity of inner and outer products, we can rearrange the formula for $\text{proj}_b$ to see



$$
\text{proj}_b(a) = \frac{b^\top a}{b^\top b}b = \frac{1}{b^\top b}bb^\top a = \frac{bb^\top}{b^\top b}a = P_ba
$$



where $P_b = \frac{bb^\top}{b^\top b}$ is an $n\times n$ matrix. 

As we mentioned before, projections should by definition satisfy the idempotence property $P^2 = P$. Let's check that this is true for $P_b$. We have



$$
P_b^2 = \frac{bb^\top}{b^\top b}\frac{bb^\top}{b^\top b} = \frac{1}{(b^\top b)^2}bb^\top bb^\top = \frac{1}{(b^\top b)^2}b(b^\top b)b^\top=\frac{b^\top b}{(b^\top b)^2}bb^\top = \frac{bb^\top}{b^\top b} = P_b
$$



Indeed, $P_b$ is idempotent. 

Let's look at some $2$-d examples of orthogonal projections. First, let's define a function `orthogonal_projection(b)` which takes in a vector $b$ and returns the projection matrix $P_b = \frac{bb^\top}{b^\top b}$.

```{code-cell}
import numpy as np

def orthogonal_projection(b):
    return np.outer(b,b)/np.dot(b,b)
```

 Now let's test this out with a vector that we'd like to project onto, say $b=\begin{pmatrix}1\\2\end{pmatrix}$. Let's visualize $b$.

```{code-cell}
import matplotlib.pyplot as plt

b = np.array([1,2])
origin = np.zeros(2)

plt.quiver(*origin, *b, label='b', scale=1, units='xy', color='blue')
plt.grid()

plt.xlim(-1,1.5)
plt.ylim(-1,2.5)
plt.gca().set_aspect('equal')
plt.legend()
plt.show()
```

Next, let's compute the projection matrix $P_b$ using the function we defined above.

```{code-cell}
Pb = orthogonal_projection(b)
```

Just to make sure we've done things correctly, let's verify that $P_b$ is idempotent, by checking that $P_b^2 = P_b$.

```{code-cell}
Pb2 = np.dot(Pb, Pb)
np.allclose(Pb2, Pb)
```

Indeed it is. Now, let's try projecting a vector, say $a = \begin{pmatrix}1\\ 1\end{pmatrix}$, onto $b$. 

```{code-cell}
a = np.array([1, 1])
proj_b_a = np.dot(Pb, a) # compute the projection of a onto b
residual = a - proj_b_a

plt.quiver(*origin, *b, label='b', scale=1, units='xy', color='blue')
plt.quiver(*origin, *a, label='a', scale=1, units='xy', color='green')
plt.quiver(*origin, *proj_b_a, label='proj_b(a)', scale=1, units='xy', color='red')
plt.quiver(*proj_b_a, *residual, label='a - proj_b(a)', scale=1, units='xy', color='orange')
plt.grid()

plt.xlim(-1,1.5)
plt.ylim(-1,2.5)
plt.gca().set_aspect('equal')
plt.legend(loc='upper left')
plt.show()
```

This plot now largely replicates the figure we saw earlier: we see that 1) the projection of $a$ onto $b$ is a vector (in red) which is parallel to $b$ and 2) the residual $a - \text{proj}_b(a)$ is at a $90^\circ$ angle from $\text{proj}_b(a)$. 

### Oblique projections onto a vector

While orthogonal projections are commonly used, and in many ways special, they are not the only way we can project onto a vector. Indeed, we can define a projection of a vector $a$ onto another vector $b$ not just along the direction orthogonal to $b$, but along any arbitrary direction. The projection of a vector $a$ onto the vector $b$ _along the direction $c$_ is given by the following:



$$
\text{proj}_{b,c}(a) = \frac{c^\top a}{b^\top c}b
$$



Again, $\frac{c^\top a}{b^\top c}$ is just a scalar, so this vector is again just a rescaled version of $b$. We can also rearrange this formula to write it as a linear function in terms of a matrix



$$
P_{b,c} = \frac{bc^\top}{b^\top c}
$$



So that $\text{proj}_{b,c}(a) = P_{b,c}a$. Let's verify that $P_{b,c}$ also satisfies the idempotence property $P^2 = P$. 



$$
P_{b,c}^2 = \frac{bc^\top}{b^\top c}\frac{bc^\top}{b^\top c} = \frac{1}{(b^\top c)^2}bc^\top bc^\top = \frac{1}{(b^\top c)^2}b(c^\top b)c^\top = \frac{c^\top b}{(b^\top c)^2}bc^\top = \frac{bc^\top}{b^\top c} = P_{b,c}.
$$



Indeed, $P_{b,c}$ is also a valid projection. So then what is the difference between $P_{b,c}$ and the orthogonal projection $P_b$ that we saw before? The difference lies in the fact that the _residuals_ are no longer orthogonal; that is, the angle between $\text{proj}_{b,c}(a)$ and $a - \text{proj}_{b,c}(a)$ is no longer $90^\circ$. Let's check that this.



$$
\text{proj}_{b,c}(a)^\top (a - \text{proj}_{b,c}(a)) = \left(\frac{c^\top a}{b^\top c}b\right)^\top\left(a - \frac{c^\top a}{b^\top c}b\right) = \frac{c^\top a}{b^\top c}\left(b^\top a - \frac{c^\top a}{b^\top c}b^\top b\right)
$$



This quantity will only be zero for any $a$ if  $b^\top a - \frac{c^\top a}{b^\top c}b^\top b = 0$. This happens when $b=c$, in which case we return to get the orthogonal projection back, but for any other $c$ we will not have that the residuals are orthogonal to $\text{proj}_{b,c}(a)$. Therefore, projections of this form are called _oblique_ projections. Let's see an example in $\mathbb{R}^2$. 

Let's write a function to compute $\text{proj}_{b,c}$. 

```{code-cell}
def oblique_projection(b, c):
    return np.outer(b,c)/np.dot(b,c)
```

Let's again project the vector $b=\begin{pmatrix}1\\1\end{pmatrix}$ onto the vector $b=\begin{pmatrix}1\\2\end{pmatrix}$, but this time along the direction $c =\begin{pmatrix}1\\1/4\end{pmatrix}$. First, we'll compute $P_{b,c}$ using our oblique projection function.

```{code-cell}
c = np.array([1,0.25])

Pbc = oblique_projection(b,c)
```

Let's verify that $P_{b,c}^2 = P_{b,c}$. 

```{code-cell}
Pbc2 = np.dot(Pbc, Pbc)

np.allclose(Pbc2, Pbc)
```

So $P_{b,c}$ is indeed idempotent. Now let's visualize $a,b, \text{proj}_{b,c}(a)$ and $a-\text{proj}_{b,c}(a)$. 

```{code-cell}
proj_bc_a = np.dot(Pbc, a) # compute the projection of a onto b
residual = a - proj_bc_a

plt.quiver(*origin, *b, label='b', scale=1, units='xy', color='blue')
plt.quiver(*origin, *a, label='a', scale=1, units='xy', color='green')
plt.quiver(*origin, *proj_bc_a, label='proj_bc(a)', scale=1, units='xy', color='red')
plt.quiver(*proj_bc_a, *residual, label='a - proj_bc(a)', scale=1, units='xy', color='orange')
plt.grid()

plt.xlim(-1,1.5)
plt.ylim(-1,2.5)
plt.gca().set_aspect('equal')
plt.legend(loc='upper left')
plt.show()
```

In this plot, we see that $\text{proj}_{b,c}(a)$ is indeed parallel to $b$, but it is not at a $90^\circ$ angle from the residual $a-\text{proj}_{b,c}(a)$. Let's actually compute the angle between $\text{proj}_{b,c}(a)$ and $a-\text{proj}_{b,c}(a)$ using techniques that we learned earlier in the chapter (recall that the angle between vectors $x$ and $y$ is $\arccos\left(\frac{x^\top y}{\|x\|_2\|y\|_2}\right)$). 

```{code-cell}
temp = np.dot(proj_bc_a, residual)
residual_norm = np.linalg.norm(residual)
proj_bc_a_norm = np.linalg.norm(proj_bc_a)

angle = np.arccos(temp/(residual_norm*proj_bc_a_norm))
angle
```

Indeed, we get that the angle between $\text{proj}_{b,c}(a)$ and $a-\text{proj}_{b,c}(a)$ is approximately $2.43$ radians, which is roughly $140^\circ$. 

## Projections onto a subspace

In the above sections, we phrased projections as projecting a vector $a$ onto another _vector_ $b$. In reality, what we computed was actually the projection of $a$ onto the _subspace_ $V = \text{span}(b)$. 

It turns out that there's nothing special about projecting onto a $1$-dimensional subspace: we can define orthogonal and oblique projections onto any subspace, as we will see below.

### Orthogonal projections onto a subspace

Let's begin with the concept of an orthogonal projection onto a subspace. Let $V$ be a subspace of $\mathbb{R}^n$, spanned by vectors $a_1,\dots, a_k$. Let's let $A$ be the matrix whose columns are $a_1,\dots, a_k$, i.e.



$$
A = \begin{pmatrix} | & | && |\\ a_1 & a_2 & \cdots & a_k \\ | & | & & |\end{pmatrix}
$$



Let's see how we can derive the orthogonal projection onto $V$, which is just the column space of $A$. Consider projecting a vector $b$ onto $V$ -- call this projection $\hat{b} = \text{proj}_V(b)$. Since $\hat{b}$ should belong to the column space of $A$, it should be of the form $\hat{b} = A\hat{x}$ for some vector $\hat{x}.$ For this projection to be orthogonal, we want that $b- \hat{b} = b-A\hat{x}$ to be orthogonal to all the columns of $A$. Earlier in the chapter, we saw that this means that $A^\top(b - A\hat{x}) = 0$. Then



$$
A^\top(b-A \hat{x}) = 0 \iff A^\top b = A^\top A\hat{x} \iff \hat{x} = (A^\top A)^{-1}A^\top b
$$



Since $\hat{b} = A\hat{x}$, we get



$$
\text{proj}_V(b) = \hat{b} = A\hat{x} = A(A^\top A)^{-1}A^\top b
$$



This immediately gives us a formula for the projection matrix $P_V$: 



$$
P_V = A(A^\top A)^{-1}A^\top
$$



This is an important formula that we will see again later in the semester. Let's check that $P_V$ satisfies the idempotence condition $P_V^2 = P_V$. We have



$$
P_V^2 = A(A^\top A)^{-1}\underbrace{A^\top A(A^\top A)^{-1}}_{I}A^\top = A(A^\top A)^{-1}A^\top = P_V
$$



Indeed it does. Now let's look at an example numerically. Consider the matrix



$$
A =\begin{pmatrix} a_1 & a_2\end{pmatrix} = \begin{pmatrix} 1 & 0\\ 1 & 1 \\ 0 &1\end{pmatrix}
$$



Where here $a_1 = \begin{pmatrix}1\\1\\0\end{pmatrix}$ and $a_2 = \begin{pmatrix}0\\ 1\\1\end{pmatrix}$. Let's compute the projection onto $V = \text{span}(a_1,a_2)$.



```{code-cell}
A = np.array([[1,0], [1,1], [0,1]])
ATA_inv = np.linalg.inv(np.dot(A.T, A)) # compute (A^TA)^{-1}
PV = np.dot(A, np.dot(ATA_inv, A.T))
```

Let's verify that this worked, by checking numerically that $P_V^2 = P_V$. 

```{code-cell}
PV2 = np.dot(PV, PV)
np.allclose(PV2, PV)
```

Indeed it does.

Let's now verify that this projection is orthogonal to the column space of $A$, by computing $A^\top(b - \text{proj}_V(b))$. For this example, we'll use the vector $b = \begin{pmatrix}1\\ 2\\ 3\end{pmatrix}$. 

```{code-cell}
b = np.array([1,2,3])

proj_V_b = np.dot(PV, b)
residual = b - proj_V_b

np.dot(A.T, residual).round(8)
```

Indeed, we get the zeros vector, and so $b-\text{proj}_V(b)$ is orthogonal to the columns of $A$. 

#### Relationship with the QR decomposition 

In the previous workbook, we saw that we can write any matrix $A$ as $A = QR$ where $Q$ is an orthogonal matrix and $R$ is upper triangular. Here, we'll see that we can write the projection onto the column space conveniently in terms of $Q$. Let's plug in $A = QR$ into our formula for $P_V$ (and recall that $Q^\top Q = I$).



$$
P_V = QR((QR)^\top QR)^{-1}(QR)^\top = QR(R^\top \underbrace{Q^\top Q}_{I} R)^{-1}R^\top Q^\top = QR(R^\top R)^{-1}R^\top Q^\top = Q\underbrace{RR^{-1}}_{I}\underbrace{(R^\top)^{-1}R^\top}_{I} Q^\top = QQ^\top
$$



Therefore, if we have the QR decomposition of $A$, the projection onto the column space of $A$ can be easily computed with $QQ^\top$. This is convenient as it doesn't require taking any matrix inverses, which can be difficult to work with numerically.

**Remark:** Recall that we always have that $Q^\top Q = I$ for an orthogonal matrix $Q$. Here we see clearly that $QQ^\top$ is emphatically _not_ equal to the identity in general.

Let's use this method to compute the projection $P_V$ using the same matrix $A$ as above. Here we use the built-in numpy function for the QR decomposition, but we could just as well have used the QR function that we wrote ourselves in the previous workbook.

```{code-cell}
Q, R = np.linalg.qr(A)
QQT = np.dot(Q, Q.T)
np.allclose(QQT, PV)
```

Indeed, the two approaches give us the same answer.

The last point we make before moving on is that there are _many_ possible matrices $A$ whose columns span a given subspace $V$. For example, the matrix



$$
B = \begin{pmatrix} -2 & 0\\ -2 & 4 \\ 0 &4\end{pmatrix}
$$



has the same column space as $A$. Let's check that computing the projection using this matrix gives us the same result.

```{code-cell}
B = np.array([[-2, 0], [-2, 4], [0, 4]])

Q2, R2 = np.linalg.qr(B)
QQT2 = np.dot(Q2, Q2.T)
np.allclose(QQT, QQT2)
```

Indeed, the projection onto $V$ is the same no matter which spanning vectors we use.

### Oblique projections onto a subspace

Like in the case of projecting onto vectors, we can also have _oblique_ projections onto the column space of a matrix $A$, which is the subspace $V$. Let $C$ be any $n\times k$ matrix such that $C^\top A$ is invertible. Then the matrix



$$
P_{V,C} = A(C^\top A)^{-1}C^\top
$$



is always a projection onto $V$. It will of course reduce to the orthogonal projection when $C = A$, in which case we obtain the same  formula that we had before. To check that it is indeed a projection, we need to verify that $P_{V,C}^2 = P_{V,C}$. We calculate



$$
P_{V,C}^2 = A(C^\top A)^{-1}\underbrace{C^\top A(C^\top A)^{-1}}_{I}C^\top = A(C^\top A)^{-1}C^\top = P_{V,C}
$$


So $P_{V,C}$ is in fact a valid projection. Let's first look at an example with the matrix $A = \begin{pmatrix} 1 & 0\\ 1 & 1 \\ 0 &1\end{pmatrix}$ that we used above. There are many valid examples of matrices $C$ that we can use to define an oblique projection $P_{V,C}$; indeed, for most matrices $C$ we will have that $C^\top A$ is invertible. Let's try choosing $C$ to be a random matrix.

```{code-cell}
k = 2
n = 3

C = np.random.normal(size = (n,k))
```

Let's check that $C^\top A$ is invertible.

```{code-cell}
CTA_inv = np.linalg.inv(np.dot(C.T, A))
```

Indeed, computing the inverse works without error.

Now, let's use this to compute $P_{V,C}$. 

```{code-cell}
PVC = np.dot(A, np.dot(CTA_inv, C.T))
```

We can check numerically that $P_{V,C}^2 = P_{V,C}$:

```{code-cell}
PVC2 = np.dot(PVC, PVC)
np.allclose(PVC2, PVC)
```

So $P_{V,C}$ is in fact idempotent, and thus a valid projection. However, it is not orthogonal; we can check this by computing $A^\top (b - P_{V,C}b)$, and verifying that it is not equal to zero (as it was in the orthogonal case). Let's do this for the same vector $b = \begin{pmatrix}1\\ 2\\ 3\end{pmatrix}$ that we used before.

```{code-cell}
proj_VC_b = np.dot(PVC, b)
residuals = b - proj_VC_b
np.dot(A.T, residuals)
```

 Our answer is clearly not zero, and so the projection $P_{V,C}$ is _not_ an orthogonal projection, but rather an _oblique_ projection.

## Projecting onto the orthogonal complement of a subspace

The last type of projection we will discuss is the projection onto the _orthogonal complement_ of a subspace $V\subseteq \mathbb{R}^n$. The orthogonal complement is the subspace $V^\perp$ which is defined as follows:


$$
V^\perp = \{w\in \mathbb{R}^n : w^\top v = 0\text{ for all } v\in V\}
$$


That is, the orthogonal complement of $V$ is the set of all vectors which are orthogonal to all vectors in $V$. It turns out that the projection onto the orthogonal complement is easy to find given the orthogonal projection onto $V$. If $P_V$ is the orthogonal projection onto $V$, then the orthogonal projection onto $V^\perp$ is just


$$
P_{V^\perp} = I - P_V
$$
 

Given $P_V = A(A^\top A)^{-1}A^\top$ or $P_V = QQ^\top$ (where $Q$ comes from the $QR$ factorization of $A$), this means $P_{V^\perp} = I- A(A^\top A)^{-1}A^\top$ or $P_{V^\perp} = I- QQ^\top$. Since the range of $P_V$ is $V$, and the range of $P_{V^\perp}$ is $V^\perp$, we should always have that $P_V x$ is orthogonal to $P_{V^\perp}y$ for any vectors $x$ and $y$. Let's check that this is in fact true. We have


$$
(P_{V^\perp}y)^\top P_{V}x = y^\top P_{V^\perp}P_Vx = y^\top (I-P_V)P_Vx = y^\top P_Vx - y^\top P_V^2x = y^\top P_Vx - y^\top P_V x = 0
$$


where we used the fact that $P_V$ is a projection, so $P_V^2 = P_V$. 