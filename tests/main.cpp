#include "../siclib/siclib.hpp"
#include <pybind11/embed.h>
#include <iostream>
int main()
{
	pybind11::scoped_interpreter guard{};

	{
		std::vector<size_t> shape = { 1, 2, 5 };
		sic::TensorView my_view = sic::generate_tensor(shape, 0);
	}

	std::vector<double> my_vec = { 0, 1, 2, 3, 4, 5, 0, 2, 4, 6, 8, 10 };
	sic::TensorView my_view{ my_vec, {2, 3, 2} };
	std::cerr << my_view << "\n";
	auto v = my_view.get_shape();

	auto result = sic::find_moment(my_view, 1, 1);
	auto k = result.get_buffer();
	std::cerr << result;

	return 1;
}
