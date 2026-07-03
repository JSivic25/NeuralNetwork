#include "layer.h"

layer::layer(int numInputs, int numNeurons, std::string activationFunction) {
	biases = std::vector<double>(numNeurons, 0.0);
	weights = std::vector<std::vector<double>>(numNeurons, std::vector<double>(numInputs, 0.0)); //weights[nueron][input]
	this->numInputs = numInputs;
	this->numNeurons = numNeurons;
	this->activationFunction = activationFunction;
}

void layer::loadWeights(const std::vector<std::vector<double>>& weights) {
	this->weights = weights;
}
void layer::loadBiases(const std::vector<double>& biases) {
	this->biases = biases;
}

const std::vector<std::vector<double>>& layer::getWeights() const {
	return weights;
}
const std::vector<double>& layer::getBiases() const {
	return biases;
}

std::vector<double> layer::forwardPass(const std::vector<double>& inputs) {
	return std::vector<double>(numNeurons, 0.0);
}
std::vector<double> layer::backProp(const std::vector<double>& errorSignal, double learningRate) {
	return std::vector<double>(numNeurons, 0.0);
}