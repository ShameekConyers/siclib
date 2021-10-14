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
I have made available as a  Python package with the underlying implementation
coded in C++ and made available from the source below.

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

	#### Solution:
	We utilize the *Secant Method* which offers much better convergence
	than binary search

	#### Implementation Example
	```python
	>>> import sicnumerical, numpy
	>>> def example_func(x):
	... 	return x ** 2 + 2 * x - 1
	>>> target_y = 14
	>>> sicnumerical.find_equation_solution(example_func, target_y)
	1.0
	>>> def no_solution(x):
	... 	return x ** 2 + 1
	>>> target_y = 0
	>>> sicnumerical.find_equation_solution(no_solution, target_y)
	None
	```




- ### Differentiation

	Approximate $\dfrac{d f(x)}{d x}$ given an array $X = [(x_1, f(x_1)),
	..., (x_n, f(x_n))]$.

	#### Solution:
	We will utilize *Five-Point Midpoint Method* when $(x, f(x)) \in X$

	#### Implementation Example:
	```python
	>>> import sicnumerical, numpy
	>>> data = numpy.array([x ** 2 + x for x in range(11)])
	>>> x_value = 5
	>>> sicnumerical.find_derivative_from_index(data, x_value)
	11.0
	```

- ### Integration

	Approximate $F(x),\enspace F(x) = \int_{a}^{b}f(x)dx$
	given an array $X = [(x_1, f(x_1)),..., (x_n, f(x_n))]$.

	#### Solution:
	We will utilize "Composite Simpson's Rule"

	#### Implementation Example:
	```python
	>>> import sicnumerical, numpy
	>>> data = numpy.array([x ** 2 + x for x in range(11)])
	>>> int_start, int_end = 6, 11
	... #note the only the interval [start, end - 1] is integrated over
	>>> sicnumerical.find_integral_from_index(data, int_start, int_end)
	279.3333
	```

<!--
### Interpolation -->
