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

	previousInput = inputs;

	std::vector<double> weightedSums = computeWeightedSums(inputs); 

	std::vector<double> activatedOutput = applyActivationFunction(weightedSums);

	return activatedOutput;
}

std::vector<double> layer::backProp(const std::vector<double>& errorSignal, double learningRate) {

	std::vector<std::vector<double>> partialLossWRTWeight = computeWeightGradients(errorSignal);
	std::vector<double> partialLossWRTBias = errorSignal;

	std::vector<double> nextErrorSignal = computeNextErrorSignal(errorSignal);

	updateWeights(partialLossWRTWeight, learningRate);
	updateBiases(partialLossWRTBias, learningRate);

	return nextErrorSignal;
}

//start of forward pass helper functions
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

std::vector<double> layer::applyActivationFunction(const std::vector<double>& weightedSums) {

	if (activationFunction == "relu") {
		return reluActivation(weightedSums);
	}
}

std::vector<double> layer::reluActivation(const std::vector<double>& weightedSums)
{
	std::vector<double> activatedOutput(numNeurons, 0.0);

	for (int i = 0; i < numNeurons; i++) {
		activatedOutput[i] = std::max(0.0, weightedSums[i]);
	}

	return activatedOutput;
}

//start of backprop helper functions
std::vector<std::vector<double>> layer::computeWeightGradients(const std::vector<double> errorSignal) {
	std::vector<std::vector<double>> weightGradients(numNeurons, std::vector<double>(numInputs, 0.0));
	for (int i = 0; i < weights.size(); i++) {
		for (int j = 0; j < weights[0].size(); j++) {
			weightGradients[i][j] = errorSignal[i] * previousInput[j];
		}
	}

	return weightGradients;
}

std::vector<double> layer::computeNextErrorSignal(const std::vector<double> previousErrorSignal) {
	std::vector<double> newErrorSignal(numInputs);

	for (int i = 0; i < weights[0].size(); i++) {
		double sum = 0;
		for (int j = 0; j < weights.size(); j++) {
			sum += previousErrorSignal[j] * weights[j][i];
		}
		newErrorSignal[i] = sum;
	}

	return newErrorSignal;
}

void layer::updateWeights(std::vector<std::vector<double>> weightGradients, double learningRate) {
	std::vector<std::vector<double>> newWeights = getWeights();

	for (int i = 0; i < weightGradients.size(); i++) {
		for (int j = 0; j < weightGradients[0].size(); j++) {
			newWeights[i][j] = newWeights[i][j] - weightGradients[i][j] * learningRate;
		}
	}

	loadWeights(newWeights);
}

void layer::updateBiases(std::vector<double> biasGradient, double learningRate) {
	std::vector<double> newBiases = getBiases();

	for (int i = 0; i < biasGradient.size(); i++) {
		newBiases[i] = newBiases[i] - biasGradient[i] * learningRate;
	}

	loadBiases(newBiases);
}