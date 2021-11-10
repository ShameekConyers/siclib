#pragma once

#include <initializer_list>
#include <exception>
#include <algorithm>
#include <vector>

namespace sic
{

template<typename T, size_t K = 10>
struct small_vector {
	using value_type = T;
	using iterator = T*;
	using const_iterator = const T*;

	small_vector()
	{
		m_size = 0;
	}

	small_vector(std::initializer_list<T> input)
	{

		m_size = input.size();
		if (m_size > K) {
			throw std::runtime_error("Too big");
		}

		std::copy(std::begin(input), std::end(input), std::begin(m_data));
	}

	template<template <typename ...> class Container, typename U>
	small_vector(const Container<U>& input)
	{
		m_size = input.size();
		if (m_size > K) {
			throw std::runtime_error("Too big");
		}
		std::copy(std::begin(input), std::end(input), std::begin(m_data));
	}

	operator std::vector<T>() const
	{
		std::vector<T> result(cbegin(), cend());
		return result;
	}

	iterator begin()
	{
		return &m_data[0];
	}

	iterator end()
	{
		return m_data + m_size;
	}

	const_iterator begin() const
	{
		return &m_data[0];
	}

	const_iterator end() const
	{
		return m_data + m_size;
	}
	const_iterator cbegin() const
	{
		return &m_data[0];
	}

	const_iterator cend() const
	{
		return m_data + m_size;
	}

	T& operator[](size_t index)
	{
		return m_data[index];
	}

	const T& operator[](size_t index) const
	{
		return m_data[index];
	}

	bool operator==(const small_vector<T>& other) const
	{
		if (size() != other.size()) {
			return false;
		}
		for (size_t i = 0; i < m_size; i++) {
			if (other[i] != (*this)[i]) return false;
		}
		return true;
	}

	void push_back(T item)
	{
		if (m_size >= K) {
			throw std::runtime_error("sm_vector is full");
		}
		m_data[m_size] = item;
		m_size++;

	}

	void erase(iterator input)
	{
		;
	}

	size_t size() const
	{
		return m_size;
	}

	T* data()
	{
		return m_data;
	}

private:
	T m_data[K];
	size_t m_size;
};


} //
