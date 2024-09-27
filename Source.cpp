#include <iostream>
#include <vector>

using constVec = const std::vector<float>;

float w = 0.0;

float forward(float x) {
	w = sqrtf(x);
	return w;
}

std::vector<float> forward(constVec& x) {
	std::vector<float> y_pred(std::size(x));
	for (int i{ 0 }; i < std::size(y_pred); ++i) {
		y_pred[i] = forward(x[i]);
	}
	return y_pred;
}

float MSE(constVec& y, constVec& y_pred) {
	int n = std::size(y);
	float result = 0.;
	for (int i = 0; i < n; ++i) {
		result += std::pow((y_pred[i] - y[i]), 2);
	}
	return result / float(n);
}

float loss(constVec& y, constVec& y_pred) {
	return MSE(y, y_pred);
}

float mean(constVec& vec) {
	float result = 0.;
	for (auto i{ 0u }; i < vec.size(); ++i) {
		result += vec[i];
	}
	return result / float(vec.size());
}

std::vector<float> dotP(constVec& vec1, constVec& vec2) {
	std::vector<float> result(vec1.size());
	for (int i{ 0 }; i < result.size(); ++i) {
		result[i] = vec1[i] * vec2[i];
	}
	return result;
}

std::vector<float> dotP(constVec& vec1, int val) {
	std::vector<float> result(vec1.size());
	for (int i{ 0 }; i < result.size(); ++i) {
		result[i] = vec1[i] * val;
	}
	return result;
}

std::vector<float> sub(constVec& vec1, constVec& vec2) {
	std::vector<float> result(vec1.size());
	for (int i{ 0 }; i < result.size(); ++i) {
		result[i] = vec1[i] - vec2[i];
	}
	return result;
}

float gradient(constVec& x, constVec& y, constVec& y_pred) {
	std::vector<float> x_(std::size(x));
	std::vector<float> y_(std::size(y));
	for (auto i{ 0u }; i < x_.size(); ++i) {
		x_[i] = x[i];
		y_[i] = y[i];
	}
	std::vector<float> y_pred_(std::size(y_pred));
	return mean(dotP(dotP(x_, 2), sub(y_pred, y)));
}

int main() {

	std::vector<float> x = { 4, 9, 16, 25, 36, 49, 64, 81, 100 };// 121, 144, 169, 196, 225, 256, 289, 324, 361, 400};
	std::vector<float> y = { 2, 3, 4, 5, 6, 7, 8, 9, 10 };// 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
	std::vector<float> y_pred;

	std::cout << "Prediction before training: f(3) = " << forward(3)<<"\n";

	//Training
	float learning_rate = 0.0001;
	unsigned n_iters = 100;
	float l = 0.0;
	float dw = 0.0;

	for (auto epoch{ 1u }; epoch <= n_iters; ++epoch) {
		y_pred = forward(x);
		l = loss(y, y_pred);
		dw = gradient(x, y, y_pred);
		w -= learning_rate * dw;
		if (epoch % 5 == 0) {
			printf("epoch %d: w = %f, dw = %f, loss = %f\n", epoch, w, dw, l);
		}
	}
	std::cout << "Prediction after training: f(1000) = " << forward(1000)<<"\n";
	std::cout << "Prediction after training: f(8) = " << forward(8)<<"\n";
	std::cout << "Prediction after training: f(16) = " << forward(16)<<"\n";
	std::cout << "Prediction after training: f(25) = " << forward(25)<<"\n";
	std::cout << "Prediction after training: f(400) = " << forward(400)<<"\n";
}