#include "layer.h"

int numInputs{};
int numNeurons{};
std::string activation_function{};
std::vector<double> biases{};
std::vector<std::vector<double>> weights{};
std::vector<double> previousInput{};

layer::layer(int numInputs, int numNeurons, std::string activation_function) {

}

void layer::loadWeights(const std::vector<std::vector<double>>& weights) {

}
void layer::loadBiases(const std::vector<double>& biases) {

}

const std::vector<std::vector<double>>& layer::getWeights() const {

}
const std::vector<double>& layer::getBiases() const {

}

std::vector<double> layer::forwardPass(const std::vector<double>& inputs) {

}
std::vector<double> layer::backProp(const std::vector<double>& errorSignal, double learningRate) {

}