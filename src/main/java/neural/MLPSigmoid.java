package neural;

public class MLPSigmoid implements NeuralNet {

	private Mat w1, w2;
	private Mat b1, b2;
	private double lr = 0.01;

	public MLPSigmoid(int inputSize, int hiddenSize, int outputSize) {
		w1 = Mats.random(hiddenSize, inputSize, -1, 1);
		w2 = Mats.random(outputSize, hiddenSize, -1, 1);
		b1 = Mats.random(hiddenSize, 1, -1, 1);
		b2 = Mats.random(outputSize, 1, -1, 1);
	}

	public Mat predict(Mat input) {
		Mat z1 = Mats.add(Mats.dot(w1, input), b1);
		Mat a1 = Mats.sigmoid(z1);
		Mat z2 = Mats.add(Mats.dot(w2, a1), b2);
		Mat a2 = Mats.sigmoid(z2);
		return a2;
	}

	public void train(Mat input, Mat target) {
		// Forward
		Mat z1 = Mats.add(Mats.dot(w1, input), b1);
		Mat a1 = Mats.sigmoid(z1);

		Mat z2 = Mats.add(Mats.dot(w2, a1), b2);
		Mat a2 = Mats.sigmoid(z2);

		// Backward pass
		Mat outputError = Mats.subtract(a2, target); // (10,1)
		Mat outputGradient = Mats.hadamard(outputError, Mats.dsigmoid(a2)); // (10,1)

		Mat w2T = Mats.transpose(w2);
		Mat hiddenError = Mats.dot(w2T, outputGradient); // (64,1)
		Mat hiddenGradient = Mats.hadamard(hiddenError, Mats.dsigmoid(a1)); // (64,1)

		// Update weights and biases
		Mat a1T = Mats.transpose(a1);
		Mat inputT = Mats.transpose(input);

		Mat dw2 = Mats.dot(outputGradient, a1T); // (10,64)
		Mat dw1 = Mats.dot(hiddenGradient, inputT); // (64,784)

		Mats.scale(dw2, lr);
		Mats.scale(dw1, lr);
		Mats.scale(outputGradient, lr);
		Mats.scale(hiddenGradient, lr);

		w2 = Mats.subtract(w2, dw2);
		w1 = Mats.subtract(w1, dw1);
		b2 = Mats.subtract(b2, outputGradient);
		b1 = Mats.subtract(b1, hiddenGradient);
	}
}