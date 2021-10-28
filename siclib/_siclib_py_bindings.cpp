#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "siclib.hpp"
#include <pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(_pysiclib, m)
{
	// numerical
	py::module_ m_numeric = m.def_submodule("numerical");
	m_numeric.def(
		"derivative_at_index", &sic::derivative_at_index);
	m_numeric.def(
		"integral_index_interval", &sic::integral_index_interval);
	m_numeric.def(
		"equation_solution", &sic::equation_solution
	);
	m_numeric.def(
		"initial_value_problem", &sic::initial_value_problem
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
			sic::TensorView&>(),
			py::arg("other_view")
		)
		.def(
			"unitary_op",
			&sic::TensorView::unitary_op
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
		.def(
			"transpose",
			&sic::TensorView::transpose,
			py::arg("dim_1") = ssize_t(-1),
			py::arg("dim_2") = ssize_t(-1)
		)
		.def(
			"fold_op",
			&sic::TensorView::fold_op
		)
		.def(
			"deep_copy",
			&sic::TensorView::deep_copy
		)
		.def(
			"squeeze",
			&sic::TensorView::squeeze,
			py::arg("target_dim") = ssize_t(-1)
		)
		.def(
			"unsqueeze",
			&sic::TensorView::unsqueeze
		)
		.def(
			"matmul",
			&sic::TensorView::matmul
		)
		.def(
			"slice_view",
			&sic::TensorView::slice_view
		)
		.def(
			"view_buffer",
			&sic::TensorView::view_buffer
		)
		.def(
			"get_item",
			&sic::TensorView::get_item
		)
		.def(
			"dotprod",
			&sic::TensorView::dotprod
		)
		.def(
			"__repr__",
			[](const sic::TensorView& tv)
			{
				std::stringstream output;
				output << tv;
				return output.str();
			}
	);



	// stats
	py::module_ stats = m.def_submodule("stats");

	stats.def(
		"find_moment",
		&sic::find_moment);
	stats.def(
		"find_mean",
		&sic::find_mean);
	stats.def(
		"find_variance",
		&sic::find_variance);
	stats.def(
		"find_stddev",
		&sic::find_stddev);
	stats.def(
		"find_skew",
		&sic::find_skew);
	stats.def(
		"find_kurtosis",
		&sic::find_kurtosis);

	stats.def(
		"rand_normal_tensor",
		&sic::rand_normal_tensor);

	stats.def(
		"rand_normal_tensor",
		&sic::rand_uniform_tensor);

	// adaptive
	py::module_ adaptive = m.def_submodule("adaptive");

	py::class_<sic::ProtoNet>(adaptive, "ProtoNet")
		.def(py::init <
			size_t,
			size_t,
			size_t,
			size_t,
			double>())
		.def(
			"run_epoch",
			&sic::ProtoNet::run_epoch
		)
		.def(
			"query_net",
			&sic::ProtoNet::query_net
		)
		.def_readonly(
			"m_weights",
			&sic::ProtoNet::m_weights
		)
		.def_readonly(
			"m_bias",
			&sic::ProtoNet::m_bias
		)
		.def_readonly(
			"m_transform",
			&sic::ProtoNet::m_transform
		)
		.def_readonly(
			"m_transform_deriv",
			&sic::ProtoNet::m_transform_deriv
		)
		;

}
