---
nav_name: Statistics
name: "numerical"
title: siclib.stats
date_added:
date_edited:
description:
---

## A Module based on Statistics.

---

## Documentation

- ### Finding A Moment

	```python
	>>> pysiclib.stats.find_moment(
	... 	input_tensor, target_dim, moment_num,
	... 	is_central, is_standardized)
	```

	#### Example:
	```python
	>>> import pysiclib
	>>> #Take in columns vectors as rows and transpose
	>>> input_tensor = pysiclib.linalg.Tensor(
	...		[[0, 1, 2],[ 3, 4, 5]]).tranpose()
	>>> print(input_tensor)

	Tensor:
	[[0, 3]
	 [1, 4]
	 [2, 5]]
	Tensor Shape: [3, 2]

	>>> # The Mean is the first Moment centered at 0
	>>> mean_tensor = pysiclib.stats.find_moment(
	... 	input_tensor, 0, 1, False, False)
	>>> print(mean_tensor)

	Tensor:
	[[1, 4]]
	Tensor Shape: [1, 2]
	```

	#### Moment Convenience functions
	- #####  Mean
	```python
	>>> pysiclib.stats.find_mean(input_tensor, target_dim)
	```
	- ##### Variance
	```python
	>>> pysiclib.stats.find_variance(input_tensor, target_dim)
	```
	- ##### Standard Deviation
	```python
	>>> pysiclib.stats.find_stddev(input_tensor, target_dim)
	```
	- ##### Skew
	```python
	>>> pysiclib.stats.find_skew(input_tensor, target_dim)
	```
	- ##### Kurtosis
	```python
	>>> pysiclib.stats.find_kurtosis(input_tensor, target_dim)
	```
