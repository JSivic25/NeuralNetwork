#include "linearAlgebra.h"
#include <iostream>

std::vector<double> linearAlgebra::addVectors(const std::vector<double>& matrixA, const std::vector<double>& matrixB) {
	if (matrixA.size() != matrixB.size()) {
		throw std::invalid_argument("Vectors must be the same size");
	}
	std::vector<double> result(matrixA.size());
	for (size_t i = 0; i < matrixA.size(); i++) {
		result[i] = matrixA[i] + matrixB[i];
	}
	return result;
}

double linearAlgebra::dotProduct(const std::vector<double>& matrixA, const std::vector<double>& matrixB) {
	if (matrixA.size() != matrixB.size()) {
		throw std::invalid_argument("Vectors must be the same size");
	}

	double sum = 0.0;

	for (int i = 0; i < matrixA.size(); i++) {
		sum += matrixA[i] * matrixB[i];
	}

	return sum;
}