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


# Sprectral Clustering

## Basics of clustering

Another useful application of the topics introduced in this chapter is a method called _spectral clustering_. Let's first define what a mean by clustering. Suppose we are given a dataset $n$ datapoints $D = \{\boldsymbol{x}_1,\dots, \boldsymbol{x}_n\} \subseteq \boldsymbol{R}^d$. Here each vector $\boldsymbol{x}_i$ corresponds to one observation, and contains $d$ features about this observation. For example, given a class with $n$ students, $\boldsymbol{x}_i$ could contain the homework grades of student $i$.

In clustering, we want to partition the dataset $D$ into $K$ subsets $C_1,\dots, C_K$ that add up to the entire set $D$. Think about this as grouping each student into one of $K$ distinct groups based on their grades. One very simple and classical example of an algorithm for this type of clustering is called $K$-means, and roughly it works as follows:

1. Randomly assign the datapoints $\boldsymbol{x}_1,\dots, \boldsymbol{x}_n$ to $K$ clusters $C_1,\dots,C_K$
2. Compute the mean of each cluster $\boldsymbol{\mu}_j = \frac{1}{|C_j|}\sum_{\boldsymbol{x}_i \in C_j}\boldsymbol{x}_i$
3. Reassign each point $\boldsymbol{x}_i$ to the cluster whose mean it is closest too
4. Repeat steps 2 and 3 until the clusters stop updating

Despite its simplicity, the $K$-means clustering algorithm is suprisingly effective, especially for relatively simple problems. However, basic $K$-means begins to perform worse when we move to higher dimensions -- i.e. when the number of features $d$ grows large. In this situation, we often want a method that utilizes some type of dimension reduction. This is where the singular value decomposition and/or eigenvalue decomposition becomes an important tool. In the next subsection, we discuss one such method that uses the EVD to make clustering easier in high dimensions.

## Spectral Clustering

Let us now see how we can use the various eigen/singular-value decompositions to define an improved clustering method in high dimensions. Given $n$ datapoints $\boldsymbol{x}_1,\dots,\boldsymbol{x}_n$, we assume we have some way to measure the similarity between two points $\boldsymbol{x},\boldsymbol{x}'$ of the form

$$
\mathsf{sim}(\boldsymbol{x},\boldsymbol{x}').
$$

A few common examples used in practice would be the cosine similarity $\mathsf{sim}(\boldsymbol{x},\boldsymbol{x}') = \frac{\boldsymbol{x^\top x'}}{\|\boldsymbol{x}\|_2\boldsymbol{x}'\|_2}$, or the so-called Gaussian RBF kernel $\mathsf{sim}(\boldsymbol{x},\boldsymbol{x}') = \exp(-\gamma \|\boldsymbol{x}-\boldsymbol{x}'\|_2)$, where $\gamma$ is a hyperparameter that we can choose as the user. Both of these metrics are larger for more "similar" datapoints, and smaller for more dissimilar points. Next, we can define the $n\times n$ similarity matrix $\boldsymbol{S}$ where

$$
\boldsymbol{S}_{ij} = \mathsf{sim}(\boldsymbol{x}_i,\boldsymbol{s}_j).
$$

Typically we assume that $\mathsf{sim}(\boldsymbol{x},\boldsymbol{x}') = \mathsf{sim}(\boldsymbol{x}',\boldsymbol{x})$ so that the matrix $\boldsymbol{S}$ is symmetric. 

Next, it is common practice to normalize this matrix in a few ways. First, we define a diagonal matrix $\boldsymbol{D} = \text{diag}(d_1,\dots, d_n)$ where

$$
d_i = \sum_{j=1}^n \boldsymbol{S}_{ij}
$$

i.e. the total amount of similarity between $\boldsymbol{x}_i$ and the rest of the datapoints. Next, we use this to normalize the similarity matrix to obtain

$$
\widetilde{\boldsymbol{S}} = \boldsymbol{D}^{-1}\boldsymbol{S}.
$$

This really amounts to dividing each column $i$ of the similarity matrix $\boldsymbol{S}$ by it's total "output" of similarity, so that they are on a consistent scale. Finally, we
