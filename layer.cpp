#include "layer.h"
#include "linearAlgebra.h"
#include <iostream>

//start of basic functions
layer::layer(int numInputs, int numNeurons, std::string activationFunction) {
	biases = std::vector<double>(numNeurons, 0.0);
	weights = std::vector<std::vector<double>>(numNeurons, std::vector<double>(numInputs, 0.0)); //weights[nueron][input]
	this->numInputs = numInputs;
	this->numNeurons = numNeurons;
	this->activationFunction = activationFunction;
}

void layer::loadWeights(const std::vector<std::vector<double>>& weights) {
	if (this->weights.size() != weights.size() || this->weights[0].size() != weights[0].size()) {
		throw std::invalid_argument("Weight dimensions do not match the layer's configuration.");
	}
	this->weights = weights;
}
void layer::loadBiases(const std::vector<double>& biases) {
	if (this->biases.size() != biases.size()) {
		throw std::invalid_argument("Bias dimensions do not match the layer's configuration.");
	}
	this->biases = biases;
}

const std::vector<std::vector<double>>& layer::getWeights() const {
	return weights;
}
const std::vector<double>& layer::getBiases() const {
	return biases;
}

//start of main functions
std::vector<double> layer::forwardPass(const std::vector<double>& inputs) {
	if (inputs.size() != numInputs) {
		throw std::invalid_argument("Input size does not match number of inputs for this layer.");
	}

	std::vector<double> weightedSums = computeWeightedSums(inputs); 

	return weightedSums;
}

std::vector<double> layer::backProp(const std::vector<double>& errorSignal, double learningRate) {
	return std::vector<double>(numNeurons, 0.0);
}

//start of helper functions
std::vector<double> layer::computeWeightedSums(const std::vector<double>& inputs) {
	{
		std::vector<double> preBiasSums(numNeurons, 0.0);
		for (int i = 0; i < numNeurons; i++) {
			preBiasSums[i] = linearAlgebra::dotProduct(inputs, weights[i]);
		}

		std::vector<double> weightedSums = linearAlgebra::addVectors(preBiasSums, biases);

		return weightedSums;
	};
}