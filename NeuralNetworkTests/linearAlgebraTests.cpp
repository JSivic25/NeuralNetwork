#include "pch.h"
#include "CppUnitTest.h"
#include "../Neural Network/linearAlgebra.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NeuralNetworkTests
{
	TEST_CLASS(NeuralNetworkTests)
	{
	public:

		TEST_METHOD(addVectorThrowsErrorOnDifferentSizes) {
			std::vector<double> a{ 1.0, 2.0, 3.0 };
			std::vector<double> b{ 4.0, 5.0 };
			Assert::ExpectException<std::invalid_argument>([&]() { linearAlgebra::addVectors(a, b); });
		}

		TEST_METHOD(addVectorsWorks) {
			std::vector<double> a{ 1.0, 2.0, 3.0 };
			std::vector<double> b{ 4.0, 5.0, 6.0 };
			std::vector<double> expected{ 5.0, 7.0, 9.0 };
			std::vector<double> result = linearAlgebra::addVectors(a, b);
			Assert::IsTrue(result == expected);
		}

		TEST_METHOD(dotProductThrowsErrorOnDifferentSizes) {
			std::vector<double> a{ 1.0, 2.0, 3.0 };
			std::vector<double> b{ 4.0, 5.0 };
			Assert::ExpectException<std::invalid_argument>([&]() { linearAlgebra::dotProduct(a, b); });
		}

		TEST_METHOD(dotProductWorks) {
			std::vector<double> a{ 1.0, 2.0, 3.0 };
			std::vector<double> b{ 4.0, 5.0, 6.0 };
			double expected{ 32.0 };
			double result = linearAlgebra::dotProduct(a, b);
			Assert::IsTrue(result == expected);
		}
	};
}
