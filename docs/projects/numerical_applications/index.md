---
nav_name: Numerical Applications
name: "numerical_analysis"
title:  Numerical Applications
date_added:
date_edited:
description:
---

## Numerical Analysis and it's Applications

This project is a collection of various Numerical Analysis techniques that
I have made coded in C++ and made available as a Python package.

## Source Code

Source Code is available <a href=https://github.com/ShameekConyers/sicnumerical/tree/main>here</a>.

## Package Documentation

Documentation can be found below

## Main Reference

I have used Numerical Analysis 9th edition by R. L. Burden, J. D. Faires
as the main reference.

---

## What is and Why use Numerical Analysis?

Numerical Analysis is the study of algorithms that use numerical approximation rather
than using symbolic manipulation to find exact solutions to solve math problems

### Example Applications

- #### Example 1
  Suppose we were in charge of managing a city where the population can be
  modeled at time $t$ by $N(t)$, a net birth rate of $\lambda$, an immigration
  rate of $v$.
	Then $N(t)$ can be modeled by the differential equation

	$$
	\dfrac{dN(t)}{dt} = \lambda N(t) + v
	$$
	Which is solved into
	$$
	N(t) = N(0)e^{\lambda t} + 	\dfrac{v}{\lambda} (e ^ {\lambda t} - 1)
	$$

	Suppose the previous year we had a population of 10,000, and after
  including the immigration of 2,000 individuals we have population 15,500 for
  the current year, giving us

	$$
	15,500 = 10,000e^{\lambda} + \dfrac{2,000}{\lambda}  (e ^ {\lambda} - 1)
	$$


	We are tasked at finding the net birth rate $\lambda$.
	The problem arises in that it is impossible to explicitly solve
	for $\lambda$, and hence we will have to use an approximation technique

- #### Example 2
  Suppose we manage a store where sales can be explained by the
	number of dollars spent on local advertising, we can then write
	$$
	Sales = f(Advertising)
	$$

  Suppose we are currently are spending $50,000 on advertising and are tasked
  with finding how sales change by the amount spent on advertising, or

	$$
	\dfrac{dSales}{dAdvertising} = f'(Advertising); \enspace Advertising = 50,000
	$$

	Given we don't know the functional form of $f(Advertising)$ we cannot
	solve for this explicitly and would have to use a numerical approximation
	technique.


## Implemented Techniques
- ### Solutions of Single Variable Equations
  Approximate $x$ such that $f(x) = y, \enspace x, y \in \R$

	#### Implementation:
	We utilize the *Secant Method* which offers much better convergence
	than binary search

	#### Implementation Example
	Solve for $x$ in
	1. $14 = x^2 + 2x - 1$
	2. $0 = x^2 + 1$

	Exact Solution:
	1. $x = 3$
	2. $x \notin \R$

	Approximation:
	```python
	>>> import sicnumerical
	>>> def example_func(x):
	... 	return x ** 2 + 2 * x - 1
	>>> target_y = 14
	>>> sicnumerical.equation_solution(example_func, target_y)
	3.0
	>>> def no_solution_example(x):
	... 	return x ** 2 + 1
	>>> target_y = 0
	>>> sicnumerical.equation_solution(no_solution_example, target_y)
	None
	```




- ### Differentiation

	Approximate $\dfrac{d f(x)}{d x}$ given an array $X = [(x_1, f(x_1)),
	..., (x_n, f(x_n))]$.

	#### Implementation:
	We utilize *Five-Point Midpoint Method* when $(x, f(x)) \in X$

	#### Implementation Example
	Solve $f'(5), \enspace f(x) = x^2 + x$

	Exact Solution: $f(5) = 11$

	Approximation:
	```python
	>>> import sicnumerical, numpy
	>>> data = numpy.array([x ** 2 + x for x in range(11)])
	>>> x_index = 5
	>>> sicnumerical.derivative_at_index(data, x_index)
	11.0
	```

- ### Integration

	Approximate $F(x),\enspace F(x) = \int_{a}^{b}f(x)dx$
	given an array $X = [(x_1, f(x_1)),..., (x_n, f(x_n))]$.

	#### Implementation:
	We utilize *Composite Simpson's Rule*

	#### Implementation Example
	Solve $I = \int_{0}^{5}(x^2 + x)dx$

	Exact Solution: $I = 325/6 = 51.1666...$

	Approximation:
	```python
	>>> import sicnumerical, numpy
	>>> unit_steps = 100
	>>> data = numpy.array(
	... 	[(x / unit_steps) ** 2 + (x / unit_steps) for x in range(5 * unit_steps)])
	>>> int_start, int_end = 0, 5 * unit_steps
	... #note the interval [start, end] is integrated over
	>>> sicnumerical.integral_index_interval(data, int_start, int_end)
	54.107366
	```

- ### Differential Equations - Initial Value problems

  For given $t \in [a, b]$ approximate $y_1(t), ..., y_n(t)$ given an
	$n$-order system of initial value problems having the form

	$$\dfrac{dy_1}{dt} = f_1(t, y_1, ..., y_n)\\
	\dfrac{dy_2}{dt} = f_2(t, y_1, ..., y_n)\\
	.\\
	.\\
	.\\
	\dfrac{dy_n}{dt} = f_n(t, y_1, ..., y_n)\\$$

	With an initial condition of $\lambda \in [a, b]$ such that

	$$y_1(\lambda) = \alpha_1\\
	y_2(\lambda) = \alpha_2\\
	.\\
	.\\
	.\\
	y_n(\lambda) = \alpha_n$$

	Where there may exist an arbitrary amount of $y_i(t) = y'_j(t),\enspace i \neq j$

	#### Implementation
	We utilize *Runge-Kutta Method*

	#### Implementation Example

	$$\begin{aligned}
	y_1' &= -4y_1 + 3y_2 + 6, \enspace &y_1(0) = 0\\
	y_2' &= 0.6y'_1 - 0.2y_2, \enspace &y_2(0) = 0
	\end{aligned}$$
	Let $Y(t) = (y_1(t), y_2(t))$, find $Y(0.5)$

	Exact Solution: $Y(0.5) = (1.793527048, 1.014415452)$

	Approximation:
	```python
	>>> import sicnumerical, numpy
	>>> def system_of_eqs(t, var_arr):
	... 	dvar_arr = numpy.array([0.0, 0.0])
	... 	var_arr[0] = -4 * var_arr[0] + 3 * var_arr[1] + 6
	... 	dvar_arr[1] = 0.6 * dvar_arr[0] - 0.2 * var_arr[1]
	... 	return dvar_arr
	>>> init_cond = np.array([0.0, 0.0])
	>>> init_val = 0.0
	>>> target_val = 0.5
	>>> sicnumerical.initial_value_problem(
	... 	system_of_eqs, init_cond, target_val, init_val)
	[1.79352705 1.01441545]

	```
