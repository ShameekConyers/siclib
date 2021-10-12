
#include <vector>
#include <functional>

class Interpolation {

	Interpolation(std::vector<double> data);

	double operator()(double input);

	std::vector<double> m_data;
};

class CubicSpline {
	CubicSpline(std::vector<double> data);

	double operator()(double input);

	std::vector<double> m_data;
};
