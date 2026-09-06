#include "pch.h"
#include "CppUnitTest.h"
#include <string>
#include "../Neural Network/layer.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NeuralNetworkTests
{
	TEST_CLASS(NeuralNetworkTests)
	{
	public:
		
		TEST_METHOD(constructorSetsBiasesCorrectly)
		{
			layer l(3, 2, "relu");
			Assert::IsTrue(l.getBiases().size() == 2);
		}

		TEST_METHOD(constructorSetsWeightsCorrectly) {
			layer l(3, 2, "relu");
			Assert::IsTrue(l.getWeights().size() == 2);
			Assert::IsTrue(l.getWeights()[0].size() == 3);
		}

		TEST_METHOD(loadWeightsWorks) {
			std::vector<std::vector<double>> testWeights{ { 1, 2, 3 }, { 4, 5, 6 } };
			layer l(3, 2, "relu");
			l.loadWeights(testWeights);
			Assert::IsTrue(l.getWeights() == testWeights);
		}

		TEST_METHOD(loadBiasesWorks) {
			std::vector<double> testBiases{ 1, 2 };
			layer l(3, 2, "relu");
			l.loadBiases(testBiases);
			Assert::IsTrue(l.getBiases() == testBiases);
		}

		TEST_METHOD(forwardPassThrowsErrorOnInnacurateInput) {
			layer l(3, 2, "relu");
			Assert::ExpectException<std::invalid_argument>([&l]() { l.forwardPass({ 1, 2, 3, 4 }); });
		}

		TEST_METHOD(forwardPassOutputsCorrectSize) {
			layer l(3, 2, "relu");
			layer l2(6, 4, "relu");
			std::vector<double> output = l.forwardPass({ 1, 2, 3 });
			std::vector<double> output2 = l2.forwardPass({ 1, 2, 3, 4, 5, 6 });
			Assert::IsTrue(output.size() == 2);
			Assert::IsTrue(output2.size() == 4);
		}

		TEST_METHOD(forwardPassWorksWithTrivialCase){
			layer l(3, 2, "relu");
			l.loadWeights({ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } });
			l.loadBiases({ 0.0, 0.0 });
			std::vector<double> output = l.forwardPass({ 0.0, 0.0, 0.0 });
			Assert::IsTrue(output == std::vector<double>{0.0, 0.0});
		}

		TEST_METHOD(forwardPassWorksWithNonTrivialWeightsAndInputs) {
			layer l(3, 2, "relu");
			l.loadWeights({ { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
			l.loadBiases({ 0.0, 0.0 });
			std::vector<double> output = l.forwardPass({ 1.0, 2.0, 3.0 });
			std::vector<double> expectedOutput{ 14.0, 32.0 };
			Assert::IsTrue(output == expectedOutput);
		}

		TEST_METHOD(forwardPassWorksWithNonTrivialBiases) {
			layer l(3, 2, "relu");
			l.loadWeights({ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } });
			l.loadBiases({ 1.0, 2.0 });
			std::vector<double> output = l.forwardPass({ 0.0, 0.0, 0.0 });
			std::vector<double> expectedOutput{ 1.0, 2.0 };
			Assert::IsTrue(output == expectedOutput);
		}

		TEST_METHOD(forwardPassWorksInGeneralCase) {
			layer l(3, 2, "relu");
			l.loadWeights({ { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
			l.loadBiases({ 1.0, 2.0 });
			std::vector<double> output = l.forwardPass({ 1.0, 2.0, 3.0 });
			std::vector<double> expectedOutput{ 15.0, 34.0 };
			Assert::IsTrue(output == expectedOutput);
		}

		TEST_METHOD(forwardPassWorksForLargerInput) {
			layer l(5, 4, "relu");
			l.loadWeights({ { 1.0, 2.0, 3.0, 4.0, 5.0 }, { 6.0, 7.0, 8.0, 9.0, 10.0 }, { 11.0, 12.0, 13.0, 14.0, 15.0 }, { 16.0, 17.0, 18.0, 19.0, 20.0 } });
			l.loadBiases({ 1.0, 2.0, 3.0, 4.0 });
			std::vector<double> output = l.forwardPass({ 1.0, 2.0, 3.0, 4.0, 5.0 });
			std::vector<double> expectedOutput{ 56.0, 132.0, 208.0, 284.0 };
			Assert::IsTrue(output == expectedOutput);
		}

		TEST_METHOD(forwardPassWorksForNegativeRELU) {
			layer l(3, 2, "relu");
			l.loadWeights({ { -1.0, -2.0, -3.0 }, { -4.0, -5.0, -6.0 } });
			l.loadBiases({ 0.0, 0.0 });
			std::vector<double> output = l.forwardPass({ 1.0, 2.0, 3.0 });
			std::vector<double> expectedOutput{ 0.0, 0.0 };
			Assert::IsTrue(output == expectedOutput);
		}

		TEST_METHOD(backpropUpdatesWeightsCorrectly) {
			layer l(1, 1, "relu");
			l.loadWeights({ { 0.5 } });
			l.loadBiases({ 0.0 });
			l.forwardPass({ 1.0 });
			l.backProp({ 1.0 }, 0.1);
			double newWeight = l.getWeights()[0][0];
			double expectedWeight = 0.4;
			Assert::IsTrue(newWeight == expectedWeight);
		}

		TEST_METHOD(backpropUpdatesBiasesCorrectly) {
			layer l(1, 1, "relu");
			l.loadWeights({ { 0.5 } });
			l.loadBiases({ 0.0 });
			l.forwardPass({ 1.0 });
			l.backProp({ 1.0 }, 0.1);
			double newBias = l.getBiases()[0];
			double expectedBias = -0.1;
			Assert::IsTrue(newBias == expectedBias);
		}

		TEST_METHOD(backpropReturnsCorrectErrorSignal) {
			layer l(1, 1, "relu");
			l.loadWeights({ { 0.5 } });
			l.loadBiases({ 0.0 });
			l.forwardPass({ 1.0 });
			std::vector<double> errorSignal = l.backProp({ 1.0 }, 0.1);
			std::vector<double> expectedErrorSignal{ 0.5 };
			Assert::IsTrue(errorSignal == expectedErrorSignal);
		}

		TEST_METHOD(backpropUpdatesMultipleWeightsCorrectly) {
			layer l(2, 2, "relu");
			l.loadWeights({ { 0.5, 0.6 }, { 0.7, 0.8 } });
			l.loadBiases({ 0.0, 0.1 });
			l.forwardPass({ 1.0, 2.0 });
			l.backProp({ 1.0, 2.0 }, 0.1);
			std::vector<std::vector<double>> newWeights = l.getWeights();
			std::vector<std::vector<double>> expectedWeights{ { 0.4, 0.4 }, { 0.5, 0.4 } };
			for (int i = 0; i < newWeights.size(); i++) {
				for (int j = 0; j < newWeights[0].size(); j++) {
					Assert::AreEqual(expectedWeights[i][j], newWeights[i][j], 1e-9);
				}
			}
		}

		TEST_METHOD(backpropUpdatesMultipleBiasesCorrectly) {
			layer l(2, 2, "relu");
			l.loadWeights({ { 0.5, 0.6 }, { 0.7, 0.8 } });
			l.loadBiases({ 0.0, 0.1 });
			l.forwardPass({ 1.0, 2.0 });
			l.backProp({ 1.0, 2.0 }, 0.1);
			std::vector<double> newBiases = l.getBiases();
			std::vector<double> expectedBiases{ -0.1, -0.1 };
			Assert::IsTrue(newBiases == expectedBiases);
		}

		TEST_METHOD(backpropReturnsCorrectErrorSignalForMultipleNeurons) {
			layer l(2, 2, "relu");
			l.loadWeights({ { 0.5, 0.6 }, { 0.7, 0.8 } });
			l.loadBiases({ 0.0, 0.1 });
			l.forwardPass({ 1.0, 2.0 });
			std::vector<double> errorSignal = l.backProp({ 1.0, 2.0 }, 0.1);
			std::vector<double> expectedErrorSignal{ 1.9, 2.2 };
			for (int i = 0; i < errorSignal.size(); i++) {
				Assert::AreEqual(errorSignal[i], expectedErrorSignal[i], 1e-9);
			}
		}
	};
}
