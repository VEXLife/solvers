# Solvers
[![Test](https://github.com/VEXLife/Solvers/actions/workflows/ci.yml/badge.svg)](https://github.com/VEXLife/Solvers/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/VEXLife/Solvers/graph/badge.svg?token=HNF0UN8A98)](https://codecov.io/gh/VEXLife/Solvers)

Here are my implementation of some useful solvers for mainly optimization problems in different programming languages. I hope you find them helpful.

## Problem definition

- Scalar equation: $f(x) = 0$
- Non-negative linear system: $\mathbf{A}\mathbf{x} = \mathbf{b}$ subject to $\mathbf{x} \succeq 0$
- Non-negative least squares: $\min_{\mathbf{x}} \frac{1}{2}\|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2$ subject to $\mathbf{x} \succeq 0$
- Non-negative quadratic programming: $\min_{\mathbf{x}} \frac{1}{2}\mathbf{x}^\top\mathbf{Q}\mathbf{x} + \mathbf{c}^\top\mathbf{x}$ subject to $\mathbf{x} \succeq 0$

## List of solvers

- [x] Scalar equation solver ([MATLAB](./MATLAB/bisection_fsolve.m))
    - Algorithm: Bisection method
    - Self-contained
- [x] Non-negative linear system solver ([MATLAB](./MATLAB/fpi_kldivergence.m))
    - Optimization object: KL-Divergence
    - Algorithm: Fixed-point iteration
    - Extra requirement: The matrix $\mathbf{A}$ must be non-negative
    - Self-contained
- [x] Non-negative linear system solver ([MATLAB](./MATLAB/fpi_lsqnonneg.m))
    - Optimization object: Least Squares
    - Algorithm: Fixed-point iteration
    - Self-contained
- [x] Non-negative linear system solver ([MATLAB](./MATLAB/gd_kldivergence.m))
    - Optimization object: KL-Divergence
    - Algorithm: Gradient Descent
    - Extra requirement: The matrix $\mathbf{A}$ must be non-negative
    - Self-contained
- [x] Non-negative linear system solver ([MATLAB](./MATLAB/pgd_lsqnonneg.m))
    - Optimization object: Least Squares
    - Algorithm: Projected Gradient Descent
    - Self-contained
- [x] Non-negative quadratic programming solver ([MATLAB](./MATLAB/pgd_quadprog.m))
    - Algorithm: Projected Gradient Descent
    - Self-contained
- [x] Non-negative quadratic programming solver ([MATLAB](./MATLAB/multipupd_quadprognonneg.m))
    - Algorithm: Multiplicative update
    - Self-contained

## Performance

![Non-negative linear system solvers](./MATLAB/figs/linear_eqn.png)
![Scalar equation solvers](./MATLAB/figs/scalar_eqn.png)