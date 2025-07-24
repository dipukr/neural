package neural;

public class MLPSigmoid implements NeuralNet {

	private Mat w1, w2;
	private Mat b1, b2;
	private double lr = 0.01;

	public MLPSigmoid(int inputSize, int hiddenSize, int outputSize) {
		w1 = MatUtils.random(hiddenSize, inputSize, -1, 1);
		w2 = MatUtils.random(outputSize, hiddenSize, -1, 1);
		b1 = MatUtils.random(hiddenSize, 1, -1, 1);
		b2 = MatUtils.random(outputSize, 1, -1, 1);
	}

	public Mat predict(Mat input) {
		Mat z1 = MatUtils.add(MatUtils.dot(w1, input), b1);
		Mat a1 = MatUtils.sigmoid(z1);
		Mat z2 = MatUtils.add(MatUtils.dot(w2, a1), b2);
		Mat a2 = MatUtils.sigmoid(z2);
		return a2;
	}

	public void train(Mat input, Mat target) {
		// Forward
		Mat z1 = MatUtils.add(MatUtils.dot(w1, input), b1);
		Mat a1 = MatUtils.sigmoid(z1);

		Mat z2 = MatUtils.add(MatUtils.dot(w2, a1), b2);
		Mat a2 = MatUtils.sigmoid(z2);

		// Backward pass
		Mat outputError = MatUtils.subtract(a2, target); // (10,1)
		Mat outputGradient = MatUtils.hadamard(outputError, MatUtils.dsigmoid(a2)); // (10,1)

		Mat w2T = MatUtils.transpose(w2);
		Mat hiddenError = MatUtils.dot(w2T, outputGradient); // (64,1)
		Mat hiddenGradient = MatUtils.hadamard(hiddenError, MatUtils.dsigmoid(a1)); // (64,1)

		// Update weights and biases
		Mat a1T = MatUtils.transpose(a1);
		Mat inputT = MatUtils.transpose(input);

		Mat dw2 = MatUtils.dot(outputGradient, a1T); // (10,64)
		Mat dw1 = MatUtils.dot(hiddenGradient, inputT); // (64,784)

		MatUtils.scale(dw2, lr);
		MatUtils.scale(dw1, lr);
		MatUtils.scale(outputGradient, lr);
		MatUtils.scale(hiddenGradient, lr);

		w2 = MatUtils.subtract(w2, dw2);
		w1 = MatUtils.subtract(w1, dw1);
		b2 = MatUtils.subtract(b2, outputGradient);
		b1 = MatUtils.subtract(b1, hiddenGradient);
	}
}