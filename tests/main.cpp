#include "../siclib/siclib.hpp"
#include <pybind11/embed.h>
#include <iostream>
#include <cassert>
int main()
{
	pybind11::scoped_interpreter guard{};

	{
		std::vector<size_t> shape = { 1, 2, 5 };
		sic::TensorView my_view = sic::generate_tensor(shape, 0);
	}

	{
		std::vector<double> my_vec = { 0, 1, 2, 3, 4, 5, 0, 2, 4, 6, 8, 10 };
		sic::TensorView my_view{ my_vec, {2, 3, 2} };
		std::cerr << my_view << "\n";
		auto v = my_view.get_shape();

		// auto result = sic::find_moment(my_view, 1, 1);
		auto result = sic::find_mean(my_view, 1);
		std::cerr << result;
	}

	{
		std::vector<double> my_vec = { 0, 1, 2, 3, 4, 5 };
		sic::TensorView my_view{ my_vec, {2, 3} };
		std::cerr << my_view;
		sic::TensorView t_view = my_view.transpose();
		std::stringstream my_s;
		std::cerr << my_view.m_storage->m_data.data() << "\n";
		std::cerr << t_view.m_storage->m_data.data();
	}
	assert(false);  // just to see std::cerr
	return 0;
}
