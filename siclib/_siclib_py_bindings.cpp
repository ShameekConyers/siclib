#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "siclib.hpp"
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(_pysiclib, m)
{
	// numerical
	py::module_ m_num = m.def_submodule("numerical");
	m_num.def(
		"derivative_at_index", &derivative_at_index);
	m_num.def(
		"derivative_at_value", &derivative_at_value);
	m_num.def(
		"integral_index_interval", &integral_index_interval);
	m_num.def(
		"equation_solution", &equation_solution
	);
	m_num.def(
		"initial_value_problem", &initial_value_problem
	);

	// linalg
	py::module_ m_linalg = m.def_submodule("linalg");
	py::class_<sic::TensorView>(m_linalg, "Tensor")
		.def(py::init <
			py::array_t<double>>(),
			py::arg("numpy_array")
		)
		.def(py::init <
			std::vector<double>,
			std::vector<size_t>,
			std::vector<size_t>,
			size_t>(),
			py::arg("input_data"),
			py::arg("input_shape") = std::vector<size_t>{},
			py::arg("input_stride") = std::vector<size_t>{},
			py::arg("offset") = 0
		)
		.def(py::init <
			sic::TensorView&,
			std::vector<size_t>,
			std::vector<size_t>,
			size_t>(),
			py::arg("other_view"),
			py::arg("input_shape") = std::vector<size_t>{},
			py::arg("input_stride") = std::vector<size_t>{},
			py::arg("offset") = 0
		)
		.def(
			"binary_element_wise_op",
			&sic::TensorView::binary_element_wise_op
		)
		.def(
			"get_buffer",
			&sic::TensorView::get_buffer
		)
		.def(
			"get_shape", &sic::TensorView::get_shape
		)
		.def(
			"get_stride", &sic::TensorView::get_stride
		)
		.def(
			"get_offset", &sic::TensorView::get_offset
		)
		.def(
			"to_numpy",
			&sic::TensorView::to_numpy
		)
		;

}
