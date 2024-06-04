
# Mathematical Notation

I have tried to keep the mathematical content of the book to the minimum necessary to achieve a proper understanding of the field. However, this minimum level is nonzero, and it should be emphasized that <b>a good grasp of calculus, linear algebra, and probability theory is <u>essential</u></b> for a clear understanding of modern pattern recognition and machine learning techniques. Nevertheless, the emphasis in this book is on conveying the underlying concepts rather than on mathematical rigour.

It is very difficult to come up with a single, consistent notation to cover the wide variety of data, models and algorithms that we discuss in this book. Furthermore, conventions differ between different fields (such as machine learning, statistics and optimization), and between different books and papers within the same field. Nevertheless, we have tried to be as consistent as possible. Below we summarize most of the notation used in this book, although individual sections may introduce new notation. Note also that the same symbol may have different meanings depending on the context, although we try to avoid this where possible.

## Common mathematical symbols

We list some common symbols below.

| Symbol | Meaning | 
| --- | --- | 
| $\infty$ | Infinity |
| $\rightarrow$ | Tends towards, e.g. $n \rightarrow \infty$ |
| $\mathbb{N}$ | Natural numbers e.g. 1, 2, 3, ... |
| $\mathbb{R}^+$ | Positive real numbers e.g. 1.2, 3.14, 5.0 |
| $ \propto $ | Proportional to, so y = ax can be written as y ∝ x
| $\triangleq$ | Defined as |
| $O(\cdot)$ |  Big-O: roughly means order of magnitude |
| $\mathbb{Z}_{+}$ | The positive integers |
| $\mathbb{R}$ | Real numbers e.g. 1.2, 3.14, 5.0 |
| $\mathbb{R}_{+}$ | Positive real numbers e.g. 1.2, 3.14, 5.0 |
| $\mathcal{S_K}$ | | The K-dimensional probability simplex |
| $\mathcal{S^D_{++}}$ | Cone of positive definite $D \times D$ matrices 
| $\approx$ | Approximately equal to |
| ${ 1, \cdots, N}$ | The finite set ${1,2,...,N}$ |
| $ 1 : N $ | The finite set ${1,2,...,N}$ |
| $[l, u]$| The continuous interval ${l \lt x \lt u}$ |.



The notation $[a, b]$ is used to denote the closed interval from $a$ to $b$, that is the interval including the values $a$ and $b$ themselves, while $(a, b)$ denotes the corresponding open interval, that is the interval excluding $a$ and $b$. Similarly, $[a, b)$ denotes an interval that includes $a$ but excludes $b$. For the most part, however, there will be little need to dwell on such refinements as whether the end points of an interval are included or not.

The $M \times M$ identity matrix (also known as the unit matrix) is denoted $\mathbf{I}_M$ , which will be abbreviated to $\mathbf{I}$ where there is no ambiguity about it dimensionality. It has elements $I_{ij}$ that equal 1 if $i=j$ and 0 if $i \neq j$.

A functional is denoted $f[y]$ where $y(x)$ is some function. The concept of a functional is discussed in Appendix D.

The notation $g(x) = O(f(x))$ denotes that $|f(x)/g(x)|$ is bounded as $x → ∞$. For instance if $g(x) = 3x^2 + 2$, then $g(x) = O(x^2)$.
The expectation of a function $f(x, y)$ with respect to a random variable $x$ is denoted by $\mathbb{E}_x[f(x,y)]$. In situations where there is no ambiguity as to which variable is being averaged over, this will be simplified by omitting the suffix, for instance $\mathbb{E}[x]$. If the distribution of $x$ is conditioned on another variable $z$, then the corresponding conditional expectation will be written $\mathbb{E}_x[f(x)|z]$. Similarly, the variance is denoted $var[f(x)]$, and for vector variables the covariance is written $cov[\mathbf{x}, \mathbf{y}]$. We shall also use $cov[\mathbf{x}]$ as a shorthand notation for $cov[\mathbf{x}, \mathbf{x}]$. The concepts of expectations and covariances are introduced in Section 1.2.2.

If we have $N$ values $\mathbf{x}_1,...,\mathbf{x}_N$ of a $D$-dimensional vector $\mathbf{x} = (x_1,...,x_D)^T$, we can combine the observations into a data matrix $\mathbf{X}$ in which the $n^{th}$ row of $\mathbf{X}$ corresponds to the row vector $\mathbf{x}^T_n$ . Thus the $n, i$ element of $\mathbf{X}$ corresponds to the ith element of the $n^{th}$ observation $x_n$. For the case of one-dimensional variables we shall denote such a matrix by $\textbf{x}$, which is a column vector whose $n^{th}$ element is $x_n$. Note that $\textbf{x}$ (which has dimensionality $N$) uses a different typeface to distinguish it from $\mathbf{x}$ (which has dimensionality $D$).

## Functions 

Generic functions will be denoted by $f$ (and sometimes $g$ or $h$). We will encounter many named functions, such as $tanh(x)$ or $\omega(x)$. A scalar function applied to a vector is assumed to be applied ele- mentwise, e.g., $\mathbf{x}^2 = [x^2_1,...,x^2_D]$. Functionals (functions of a function) are written using “blackboard” font, e.g., $\mathbb{H}(p)$ for the entropy of a distribution $p$. A function parameterized by fixed parameters $\mathbf{θ}$ will be denoted by $f(x;θ)$ or sometimes $fθ(x)$. We list some common functions (with no free parameters) below.

### Common functions of 1 argument

| Symbol | Meaning | 
| --- | --- |
| ⌊x⌋ | Floor of x, i.e., round down to nearest integer |
| ⌈x⌉ | Ceiling of x, i.e., round up to nearest integer |
| ¬a | logical NOT |
| I (x) | Indicator function, I (x) = 1 if x is true, else I (x) = 0 |
| δ(x) | Dirac delta function, δ(x) = ∞ if x = 0, else δ(x) = 0  |
| $\vert x \vert$ |  Absolute value |
| $\vert \mathcal{S} \vert $ | Size (cardinality) of a set |
| n! | Factorial function |
| log(x) | Natural logarithm of x |
| exp(x) | Exponential function e^x  |
| Γ(x) | Gamma function, Γ(x) = 0 |
| Ψ(x) | Digamma function, Ψ(x) = d |
| σ(x) |  Sigmoid (logistic) function, dx 1−x |

### Common functions of 2 arguments

| Symbol | Meaning | 
| --- | --- |
| a∧b | logical AND | 
| a∨b | logical OR | 
| B(a,b) | Beta function, B(a, b) = Γ(a)Γ(b) 􏱋n􏱌 Γ(a+b)
k n choose k, equal to n!/(k!(n − k)!) δij Kronecker delta, equals I (i = j) u⊙v Elementwise product of two vectors u􏲝v Convolution of two vectors

### Common functions of > 2 arguments 


Meaning



Indicator function, I (x) = 1 if x is true, else I (x) = 0 Dirac delta function, δ(x) = ∞ if x = 0, else δ(x) = 0 Absolute value
 Size (cardinality) of a set
Factorial function
Natural logarithm of x
Exponential function ex 􏰟 ∞ x−1 −u
Gamma function, Γ(x) = 0
Digamma function, Ψ(x) = d
u e du log Γ(x)
Sigmoid (logistic) function, dx 1−

## Linear Algebra

### General notation 

Vectors are denoted by lower case bold Roman letters such as $\mathbf{x}$, and all vectors are assumed to be column vectors. A superscript $T$ denotes the transpose of a matrix or vector, so that $\mathbf{x}^T$ will be a row vector. Uppercase bold roman letters, such as $\mathbf{M}$, denote matrices. The notation $(w_1, . . . , w_M)$ denotes a row vector with M elements, while the corresponding column vector is written as $w = (w_1,...,w_M)^T$.

Vectors are bold lower case letters such as $\mathbf{x}, \mathbf{w}$. Matrices are bold upper case letters, such as $\mathbf{X}, \mathbf{W}$. Scalars are non-bold lower case, such as $x, w$. 

When creating a vector from a list of $N$ scalars, we write $x = [x_1, . . . , x_N ]$; this may be a column vector or a row vector, depending on the context. (Vectors are assumed to be column vectors, unless noted otherwise.) 

When creating an M × N matrix from a list of vectors, we write $X = [x_1,...,x_N]$ if we stack along the columns, or $X = [x_1;...;x_M]$ if we stack along the rows.

### Vectors 

Here is some standard notation for vectors. (We assume u and v are both N-dimensional vectors.)

| Symbol | Meaning |
| --- | --- |
| $\textbf{u}^T\textbf{v}$ | Inner (scalar) product, $\sum^{N}_{i=1} u_i v_i$ |
| $\mathbf{u}\mathbf{v}^T$ | Outer product ($N \times N$ matrix) |
| u⊙v | Elementwise product, [u1v1, . . . , uN vN ]  |
| $\mathbf{v}^T$ | Transpose of $\mathbf{v}$ |
| dim($\mathbf{v}$) | Dimensionality of v (namely N) |
| diag($\mathbf{v}$)  | Diagonal N × N matrix made from vector v |
| $\mathbf{1}$ or $\mathbf{1}_N$  | Vector of ones (of length N) |
| $\mathbf{0}$ or $\mathbf{0}_N$ | Vector of zeros (of length N) |
| $\Vert v \Vert$ = $\Vert v \Vert_2$  | Euclidean or $l_2$ norm $\sqrt{\sum_{i=1}^N v_i^2}$ |
| $\Vert v \Vert_1$ | $l_1$ norm $\sum_{i=1}^N \vert v_i \vert $ |

### Matrices

Here is some standard notation for matrices. (We assume S is a square N × N matrix, X and Y are of size M × N, and Z is of size M′ × N′.)

| Symbol | Meaning |
| --- | --- |
| $\mathbf{X}_{:,j}$ | j ’th column of matrix | 
| $\mathbf{X}_{i,:}$ |  i’th row of matrix (treated as a column vector) Xij Element (i, j ) of matrix | 
| $S≻0$ |  True iff S is a positive definite matrix | 
| $tr(S)$ |  Trace of a square matrix | 
| $det(S)$ |  Determinant of a square matrix | 
| $|S|$ |  Determinant of a square matrix | 
| $S−1$ |  Inverse of a square matrix | 
| $X†$ |  Pseudo-inverse of a matrix | 
| $XT$ |  Transpose of a matrix | 
| $diag(S)$ |  Diagonal vector extracted from square matrix IorIN Identity matrix of size N × N | 
| $X⊙Y$ |  Elementwise product | 
| $X⊗Z$ |  Kronecker product (see Section 7.2.5) | 

## Probability

## Optimization 

