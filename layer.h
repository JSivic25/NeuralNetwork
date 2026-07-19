#pragma once
#include <vector>
#include <string>

class layer
{
public:
	layer() = default;
	layer(int numInputs, int numNeurons, std::string activation_function);

	void loadWeights(const std::vector<std::vector<double>>& weights);
	void loadBiases(const std::vector<double>& biases);

	const std::vector<std::vector<double>>& getWeights() const;
	const std::vector<double>& getBiases() const;

	std::vector<double> forwardPass(const std::vector<double>& inputs);
	std::vector<double> backProp(const std::vector<double>& errorSignal, double learningRate);
private:
	int numInputs{};
	int numNeurons{};
	std::string activationFunction{};
	std::vector<double> biases{};
	std::vector<std::vector<double>> weights{};
	std::vector<double> previousInput{};

	std::vector<double> computeWeightedSums(const std::vector<double>& inputs);
	std::vector<double> applyActivationFunction(const std::vector<double>& weightedSums);
};

