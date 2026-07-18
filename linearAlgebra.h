#pragma once
#include <vector>
class linearAlgebra
{
	public:
		static std::vector<double> addVectors(const std::vector<double>& matrixA, const std::vector<double>& matrixB);
		static double dotProduct(const std::vector<double>& matrixA, const std::vector<double>& matrixB);
};

