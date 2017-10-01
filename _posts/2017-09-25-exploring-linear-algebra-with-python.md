---
layout: post
title:  "Exploring Linear Algebra with Python"
author: "Mubaris NK"
comments: true
---

## Scalars, Vectors and Matrices

Scalars, Vectors and Matrices are the basic objects in Linear Algebra

* **Scalar**: A scalar is just a number.
* **Vector**: A vector is an array of numbers and it's order is important.
* **Matrix**: A Matrix is a 2D array of numbers.

[NumPy](http://www.numpy.org/) is a python package that can be used for Linear Algebra calculations. We can use NumPy to create Vectors and Matrices in Python. You don't need any special packages to create Scalar, since it's just a number.

```python
# Import NumPy
import numpy as np

# Scalar
s = 32

# Vectors
a = np.array([1, 2, 3])
b = np.array([3, 4, 5])

# Matrices
A = np.array([
    [3, 5, 7],
    [4, 6, 8]
])
B = np.array([
    [4, 7],
    [5, 8],
    [6, 9]
])

# Size of the Matrices
print(A.shape) # (2, 3)
print(B.shape) # (3, 2)
```

## Operations

Many operations can applied on linear algebra objects.

### Addition

To add 2 vectors they should have same number of elements. And to add 2 matrices they sould have same size(shape).

```python
# Addition of vectors
c = a + b
print(c) # [4, 6, 8]

# Addition of matrices
X = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
Y = np.array([
    [7, 8, 9],
    [1, 2, 3]
])
Z = X + Y
print(Z) # [[8, 10, 12], [5, 7, 9]]
```

### Subtraction

To subtract 2 vectors they should have same number of elements and to subtract 2 matrices they should have size(shape).

```python
# Subtraction of vectors
d = a - b
print(d) # [-2, -2, -2]

# Subtraction of matrices
W = X - Y
print(W) # [[-6, -6, -6], [3, 3, 3]]
```

### Transpose of a Matrix

Transpose of a Matrix is an operator which flips a matrix over its main diagonal like a mirror image. This can be done by calling `numpy.transpose` function or `T` method in numpy.

```python
print(X.T) # [[1, 4], [2, 5], [3, 6]]
print(np.transpose(Y)) # [[7, 1], [8, 2], [9, 3]]

# Finding shape of transpose
print(X.shape) # (2, 3)
print(X.T.shape) # (3, 2)

print(Y.shape) # (2, 3)
print(np.transpose(Y).shape) # (3, 2)
```

### Multiplication

NumPy uses `numpy.dot` function for multiplication os both vectors and matrices. Matrix multiplication is **not commutative**.

```python
# Vectors
e = np.dot(a, b)
f = np.dot(b, a)
print(e) # 26
print(f) # 26

# Matrices
C = np.dot(A, B)
D = np.dot(B, A)
print(C) # [[79, 124], [94, 148]]
print(D) # [[40, 62, 84], [47, 73, 99], [54, 84, 114]]

# Size of C and D
print(C.shape) # (2, 2)
print(D.shape) # (3, 3)

# Vectors and Matrices
g = np.dot(A, b) # [64, 76]
```

### Inverse

Inverse operation only applies to sqaure matrices(matrix with same number of columns and rows). To compute inverse using numpy we need to use `numpy.linalg.inv` function.

```python
P = np.array([
    [1, 3, 3],
    [1, 4, 3],
    [1, 3, 4]
])
Pinv = np.linalg.inv(P)
print(Pinv) # [[7, -3, -3], [-1, 1, 0], [-1, 0, 1]]
```

## Types of Matrices

### Diagonal Matrix

A diagonal matrix is a square matrix which has zeros everywhere other than main diagonal.

### Scalar Matrix

A scalar matrix is a diagonal matrix whose diagonal values are same.

### Identity Matrix

An Identity Matrix is a scalar matrix whose diagonal values are 1.

### Zero Matrix

A zero matrix is matrix which has zeros everywhere.

#### Implementation

```python
# Identity Matrix of size (3, 3)
I3 = np.identity(3)
# Identity Matrix of size (2, 2)
I2 = np.identity(2)

# Zero Matrix
Q = np.zeros((3, 2)) # Size (3, 2)
R = np.zeros((5, 6)) # Size (5, 6)
```

## Determinant

Determinant of matrix is a special value that can be calculated from a square matrix.

```python
A = np.array([
    [1, 3, 3],
    [4, 5, 6],
    [7, 8, 9]
])
B = np.array([
    [34, 54],
    [67, 87]
])

adet = np.linalg.det(A)
bdet = np.linalg.det(B)

print(adet) # 6.0
print(bdet) # -660.0
```

## Resources

* [Beatiful Article by Brendan Fortuner](https://medium.com/towards-data-science/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c)
* [Khan Acedemy Tutorial](https://www.khanacademy.org/math/linear-algebra)
* [Linear Algebra Chapter - Deep Learning Book](http://www.deeplearningbook.org/contents/linear_algebra.html)
* [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)


<div id="mc_embed_signup">
<form action="//mubaris.us16.list-manage.com/subscribe/post?u=f9e9a4985cce81e89169df2bf&amp;id=3654da5463" method="post" id="mc-embedded-subscribe-form" name="mc-embedded-subscribe-form" class="validate" target="_blank" novalidate>
    <div id="mc_embed_signup_scroll">
	<label for="mce-EMAIL">Subscribe for more Awesome!</label>
	<input type="email" value="" name="EMAIL" class="email" id="mce-EMAIL" placeholder="email address" required>
    <!-- real people should not fill this in and expect good things - do not remove this or risk form bot signups-->
    <div style="position: absolute; left: -5000px;" aria-hidden="true"><input type="text" name="b_f9e9a4985cce81e89169df2bf_3654da5463" tabindex="-1" value=""></div>
    <div class="clear"><input type="submit" value="Subscribe" name="subscribe" id="mc-embedded-subscribe" class="button"></div>
    </div>
</form>
</div>