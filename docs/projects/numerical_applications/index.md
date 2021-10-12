---
nav_name: Numerical Applications
name: "numerical_analysis"
title:  Numerical Applications
date_added:
date_edited:
description:
---

## Numerical Analysis and it's Applications

In this project is a collection of various Numerical Analysis techniques that
I have coded in C++ and made available as a Python package interface from the
source below.

## Source Code

Source Code is available here.

Make sure to follow the installation instructions.

## Main Reference

I have used Numerical Analysis 9th edition by R. L. Burden, J. D. Faires
as the main reference.

---

## Why use Numerical Analysis?

Numerical is the study of algorithms that use numerical approximation rather
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
