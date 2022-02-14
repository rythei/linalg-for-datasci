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

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
```

# Matrices and Linear Functions

In the last section, we saw examples of both linear and non-linear functions. In this section, we continue this discussion, again focusing on the example of linear functions from $\mathbb{R}^2$ to $\mathbb{R}^2$, which allows us to visualize the transformations. 

## Matrix multiplication and linear functions

Here, we focus on linear functions of the form

$$
f(v) = Av .
$$

Where $A = \begin{pmatrix}a_{11} & a_{12}\\ a_{21}& a_{22}\end{pmatrix}$ is a $2\times 2$ matrix and $v = (v_1,v_2)$ is a vector in $\mathbb{R}^2$. 
When we write $Av$, we mean that we are multiplying the vector $v$ by the matrix $A$, namely:

$$
Av = \begin{pmatrix}a_{11} & a_{12}\\ a_{21}& a_{22}\end{pmatrix}\begin{pmatrix}v_1\\ v_2\end{pmatrix} = \begin{pmatrix}a_{11}v_1 + a_{12}v_2\\ a_{21}v_1 + a_{22}v_2\end{pmatrix}
$$

So $Av$ just gives us another vector in $\mathbb{R}^2$. 

It is easy to verify that matrix multiplication satisfies the properties of a linear function: namely for any scalar $\alpha \in \mathbb{R}$ and vectors $u,v\in \mathbb{R}^2$, we have

$$
A(\alpha v) = \alpha Av\\
A(u+v) = Au + Av
$$

That means that any function of the form $f(v) = Av$ is a linear function. 
In addition, _every_ linear function can be represented in this way. 
In other words _every_ linear function $f(v)$ can be written as $f(v) = Av$ for some matrix $A$. 
This is truly the heart of why matrices are important: they represent linear functions. 
While we may be tempted to think of matrices as multi-dimensional arrays storing numbers, it is important that we understand that they are really just a convenient representation for linear functions.

(A slight wrinkle on what we just said: this is true once a basis is given, e.g., the standard/canonical basis, or some other basis.  Then, a linear function can be represetned by a matrix, and vice versa.  A given linear function can be expressed by different matrices when a different basis is given.  In data science, we definitely are interested in representing the same linear function with respect to different basis, e.g., that is what PCA is all about, but let's not worry about that for now, and let's assume that we are working with the standard basis.)

Now that we understand this fact, given a linear function $f:\mathbb{R}^2 \to \mathbb{R}^2$, how can we find the corresponding matrix $A$? To do this, we need to identify the numbers $a_{11},a_{12},a_{21},a_{22}$. A convenient way of finding these numbers is by checking how $f$ acts on the standard basis vectors $e_1 = (1,0)$ and $e_2 = (0,1)$. Let's check what $Ae_1$ and $Ae_2$ give us:


$$
Ae_1 =\begin{pmatrix}a_{11} & a_{12}\\ a_{21}& a_{22}\end{pmatrix}\begin{pmatrix}1\\ 0\end{pmatrix}= \begin{pmatrix}a_{11}1 + a_{12}0\\ a_{21}1 + a_{22}0\end{pmatrix} = \begin{pmatrix}a_{11}\\ a_{21}\end{pmatrix}\\
Ae_2 =\begin{pmatrix}a_{11} & a_{12}\\ a_{21}& a_{22}\end{pmatrix}\begin{pmatrix}0\\ 1\end{pmatrix}= \begin{pmatrix}a_{11}0 + a_{12}1\\ a_{21}0 + a_{22}1\end{pmatrix} = \begin{pmatrix}a_{12}\\ a_{22}\end{pmatrix}
$$



These two vectors are exactly equal to the two columns of $A$! Therefore, as long as we know what $f(e_1)$ and $f(e_2)$ are, we can entirely identify what the entries of the matrix $A$ are.  

In what follows, we see a few important examples of how to do this, using similar examples to the ones we studied in the previous section.

### Rotating

Let's start again with an example of rotation. 

Recall the function $f(v)$ which takes any vector $v = (v_1,v_2)$ and rotates it by $\theta$ degrees. This function is given by the following:

$$
f(v_1, v_2) = \left(\cos(\theta)v_1 - \sin(\theta)v_2, \sin(\theta)v_1 + \cos(\theta)v_2\right)
$$

We implement this function below with $\theta = 45^\circ$ as the default.

```{code-cell}
def rotate(v, theta=np.pi/4):
    return np.array([np.cos(theta)*v[0] - np.sin(theta)*v[1], np.sin(theta)*v[0] + np.cos(theta)*v[1]])
```

Now let's visualize how this function acts on the standard basis vectors $e_1,e_2$, and try and deduce what the corresponding matrix associated with this linear function is.

```{code-cell}
origin = np.zeros(2)
e1 = np.array([1,0])
e2 = np.array([0,1])

plt.quiver(*origin, *e1, label='e1', color='blue', scale=1, units='xy')
plt.quiver(*origin, *e2, label='e2', color='red', scale=1, units='xy')
plt.quiver(*origin, *rotate(e1), label='f(e1)', color='green', scale=1, units='xy')
plt.quiver(*origin, *rotate(e2), label='f(e2)', color='orange',scale=1, units='xy')
plt.legend()
plt.grid()
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.gca().set_aspect('equal')
plt.show()
```

Here, the blue vector is $e_1$, the red vector is $e_2$, the green vector is $f(e_1)$, and the orange vector is $f(e_2)$. 
As we can see from the plot, we have $f(e_1) \approx (0.7, 0.7)$ and $f(e_2) \approx (-0.7, 0.7)$. Indeed, let's check:

```{code-cell}
print('f(e1) = ', rotate(e1))
print('f(e2) = ', rotate(e2))
```

Indeed, this is what we would expect: if we take the vector $(1,0)$ and rotate $45^\circ$, it is still a unit vector, but is on the $y=x$ line, and therefore must be $(1/\sqrt{2}, 1/\sqrt{2})$. Therefore, we have 


$$
f(e_1) = \begin{pmatrix}\frac{1}{\sqrt{2}}\\\frac{1}{\sqrt{2}}\end{pmatrix},\;\;\; f(e_2) = \begin{pmatrix}-\frac{1}{\sqrt{2}}\\\frac{1}{\sqrt{2}}\end{pmatrix}
$$



As we showed above, since $f$ is a linear function, we must have that $f(v) = Av$ for some matrix $A$, and the columns of $A$ are given by $f(e_1)$ and $f(e_2)$. 
Therefore, we deduce that the matrix $A$ must be


$$
A = \begin{pmatrix}\frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}\\ \frac{1}{\sqrt{2}} &\frac{1}{\sqrt{2}}\end{pmatrix}
$$



In Python, we can store this matrix as a numpy array as follows:

```{code-cell}
A = np.array([[1/np.sqrt(2),-1/np.sqrt(2)], [1/np.sqrt(2),1/np.sqrt(2)]])
A
```

Now since $f(v) = Av$, we realize that we don't actually need to use the function `rotate` to implement $f$: instead, we simply do matrix-vector multiplication to get the function! To do this, we can use the `np.dot` function. 
Let's make the same plot as before, but instead computing the rotation this way:

```{code-cell}
e1_rotated = np.dot(A, e1)
e2_rotated = np.dot(A, e2)

plt.quiver(*origin, *e1, label='e1', color='blue', scale=1, units='xy')
plt.quiver(*origin, *e2, label='e2', color='red', scale=1, units='xy')
plt.quiver(*origin, *e1_rotated, label='Ae1', color='green', scale=1, units='xy')
plt.quiver(*origin, *e2_rotated, label='Ae2', color='orange',scale=1, units='xy')
plt.legend()
plt.grid()
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.gca().set_aspect('equal')
plt.show()
```

As expected, we get the same result as before! Indeed, now that we've found the matrix $A$, we can use `np.dot(A,v)` to compute $f(v)$ for any vector $v \in \mathbb{R}^2$. Let's see a couple more examples with $u = (-.2, .5), v= (.9, -.3)$:

```{code-cell}
u = np.array([-.2, .5])
v = np.array([.9, -.3])

plt.quiver(*origin, *u, label='u', color='blue', scale=1, units='xy')
plt.quiver(*origin, *v, label='v', color='red', scale=1, units='xy')
plt.quiver(*origin, *np.dot(A,u), label='Au', color='green', scale=1, units='xy')
plt.quiver(*origin, *np.dot(A,v), label='Av', color='orange',scale=1, units='xy')
plt.legend()
plt.grid()
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.gca().set_aspect('equal')
plt.show()
```

### Stretching

Next, we again consider the function $f(v)$ which takes a vector $v = (v_1,v_2)$ and 'stretches' it by a factor of $\alpha$ along the x-axis and  $\beta$ along the y-axes. 
The function which performs this operations is given by

$$
f(v_1,v_2) = (\alpha v_1, \beta v_2)
$$

Let's implement it in a simple Python function, with $\alpha=2,\beta=1$ as the default, and see how it acts on the standard basis vectors.

```{code-cell}
def stretch(v, alpha=2, beta=1):
    return np.array([alpha*v[0], beta*v[1]])

plt.quiver(*origin, *e1, label='e1', color='blue', scale=1, units='xy')
plt.quiver(*origin, *e2, label='e2', color='red', scale=1, units='xy')
plt.quiver(*origin, *stretch(e1), label='f(e1)', color='green', scale=1, units='xy')
plt.quiver(*origin, *stretch(e2), label='f(e2)', color='orange',scale=1, units='xy')
plt.legend()
plt.grid()
plt.xlim(-2,2)
plt.ylim(-1,1)
plt.gca().set_aspect('equal')
plt.show()
```

As we can tell from the plot, we have that $f(e_1) = (2,0)$ and $f(e_2) = (0,1)$. It makes sense that $e_2$ is unchanged after applying $f$, since it is zero in the x-axis direction. Hence the function $f$ can be written as $f(v) = Bv$ where the matrix $B$ is given by


$$
B = \begin{pmatrix}2 & 0\\ 0&1\end{pmatrix}
$$


Let's define this as a numpy array.

```{code-cell}
B = np.array([[2,0], [0,1]])
B
```

Now we can check that $Bv$ indeed gives the same results as $f(v)$:

```{code-cell}
e1_stretched = np.dot(B, e1)
e2_stretched = np.dot(B, e2)

plt.quiver(*origin, *e1, label='e1', color='blue', scale=1, units='xy')
plt.quiver(*origin, *e2, label='e2', color='red', scale=1, units='xy')
plt.quiver(*origin, *e1_stretched, label='Be1', color='green', scale=1, units='xy')
plt.quiver(*origin, *e2_stretched, label='Be2', color='orange',scale=1, units='xy')
plt.legend()
plt.grid()
plt.xlim(-2,2)
plt.ylim(-1,1)
plt.gca().set_aspect('equal')
plt.show()
```

As expected, $f(v)$ and $Bv$ are the same.

### Projecting

The last example is the function


$$
f(v_1,v_2) = \left(\frac{1}{2}(v_1 + v_2), \frac{1}{2}(v_1 + v_2)\right)
$$

which takes any vector $v = (v_1,v_2)$, and maps it to an element of the set $L = \{(x,y)\mid x=y\}$, which is just the $y=x$ line in the plane. We implement it below, and plot how it acts on the standard basis vectors.

```{code-cell}
def project_to_line(v):
    return np.array([0.5*(v[0] + v[1]), 0.5*(v[0] + v[1])])

plt.quiver(*origin, *e1, label='e1', color='blue', scale=1, units='xy')
plt.quiver(*origin, *e2, label='e2', color='red', scale=1, units='xy')
plt.quiver(*origin, *project_to_line(e1), label='f(e1)', color='green', scale=1, units='xy')
plt.quiver(*origin, *project_to_line(e2), label='f(e2)', color='orange',scale=1, units='xy')
plt.legend()
plt.grid()
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.gca().set_aspect('equal')
plt.show()
```

The orange and green vectors (representing $f(e_1)$ and $f(e_2)$) are both equal to $(1/2, 1/2)$, and so the matrix $C$ for which $f(v) = Cv$ is given by


$$
C = \begin{pmatrix}\frac{1}{2} & \frac{1}{2}\\ \frac{1}{2} &\frac{1}{2}\end{pmatrix}
$$


We define this as a numpy array:

```{code-cell}
C = np.array([[1./2, 1./2], [1./2, 1./2]])
C
```

Now we can check that $Cv$ indeed gives the same results as $f(v)$:

```{code-cell}
e1_projected = np.dot(C, e1)
e2_projected = np.dot(C, e2)

plt.quiver(*origin, *e1, label='e1', color='blue', scale=1, units='xy')
plt.quiver(*origin, *e2, label='e2', color='red', scale=1, units='xy')
plt.quiver(*origin, *e1_projected, label='Ce1', color='green', scale=1, units='xy')
plt.quiver(*origin, *e2_projected, label='Ce2', color='orange',scale=1, units='xy')
plt.legend()
plt.grid()
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.gca().set_aspect('equal')
plt.show()
```

As expected, $f(v)$ and $Cv$ are the same.

## Finding the composition of linear functions with matrix multiplication

An important operation which we've discussed is function composition. When working with linear functions, represented as matrices, function composition can be represented by matrix multiplication. Specifically, if we have linear functions $f(v) = Av$ and $g(v) = Bv$, where $A, B$ are matrices, then we can compute the composition as follows:


$$
(f\circ g)(v) = f(g(v)) = f(Bv) = (AB)v
$$


where here $AB$ is the matrix product of $A$ and $B$. For $2\times 2$ matrices 


$$
A = \begin{pmatrix}a_{11} & a_{12}\\ a_{21}& a_{22}\end{pmatrix},\;\;\; B=\begin{pmatrix}b_{11} & b_{12}\\ b_{21}& b_{22}\end{pmatrix}
$$


the product $AB$ is given by


$$
AB = \begin{pmatrix}a_{11} & a_{12}\\ a_{21}& a_{22}\end{pmatrix}\begin{pmatrix}b_{11} & b_{12}\\ b_{21}& b_{22}\end{pmatrix} = \begin{pmatrix}a_{11}b_{11} + a_{12}b_{21}& a_{11}b_{12}+a_{12}b_{22}\\ a_{21}b_{11}+a_{22}b_{22}&a_{21}b_{12}+a_{22}b_{22}\end{pmatrix}
$$


which is just another $2\times 2$ matrix. Therefore, to compute the composition of two linear functions $f(v) = Av$, $g(v)=Bv$, we can first compute the matrix $AB$, and then apply this new matrix to a vector $v$. 

Let's see a few examples of doing this. To visualize compositions, we'll use the same method from the previous workbook of plotting how the linear functions act on points on the unit circle.

```{code-cell}
np.random.seed(345) #set random seed

n_points = 100
vv = np.random.normal(size = (2, n_points))
vv /= np.linalg.norm(vv, axis=0)
plt.scatter(vv[0], vv[1])
plt.axis('equal')
plt.show()
```

#### Rotating and stretching

Let's consider the matrices $A$ and $B$, defined above, which rotate a vector by $45^\circ$ and stretch a vector by a factor of 2 on the x-axis, respectively. Let's define a new matrix $AB$, representing the composition of first stretching, then applying a rotation.

```{code-cell}
AB = np.dot(A, B)
AB
```

  This gives us the matrix


$$
AB = \begin{pmatrix}\sqrt{2} & -1/\sqrt{2}\\ \sqrt{2}&1/\sqrt{2} \end{pmatrix}
$$


Let's verify visually that this does indeed first stretch the vectors on the circle, forming an ellipse, and then rotate them.

```{code-cell}
#here we sort the points by their angles from the origin so that the plot coloring looks nice
angles = np.arccos(np.dot(vv.T, np.array([1,0])).flatten())
angles_ix = np.argsort(angles)
vv = vv[:, angles_ix]

composed_vv = np.dot(AB, vv)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(vv[0], vv[1], c=np.arange(n_points))
ax2.scatter(composed_vv[0], composed_vv[1], c=np.arange(n_points))
ax1.axis('equal')
ax2.axis('equal')
ax1.set_title('vectors on unit circle')
ax2.set_title('vectors stretched, then rotated')
plt.show()
```

As we can see, this does exactly what we'd expect, and gives the same result as the composition from the previous workbook: it first stretches along the x-axis to form an ellipse, and then rotates each of the points by $45^\circ$. 

Next, let's try first rotating, and then stretching. This action is represented by the matrix $BA$, which we define below.

```{code-cell}
BA = np.dot(B, A)
BA
```

  This gives us the matrix


$$
BA = \begin{pmatrix}\sqrt{2} & -\sqrt{2}\\ 1/\sqrt{2}&1/\sqrt{2} \end{pmatrix}
$$
Note that clearly $AB \neq BA$: in general we do not have that matrix multiplication commutes. This corresponds to the fact that the composition of linear functions is not commutative. We can see this visually by plotting how $BA$ acts on points on the circle.

```{code-cell}
composed_vv2 = np.dot(BA, vv)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(vv[0], vv[1], c=np.arange(n_points))
ax2.scatter(composed_vv2[0], composed_vv2[1], c=np.arange(n_points))
ax1.axis('equal')
ax2.axis('equal')
ax1.set_title('vectors on unit circle')
ax2.set_title('vectors rotated, then stretched')
plt.show()
```

