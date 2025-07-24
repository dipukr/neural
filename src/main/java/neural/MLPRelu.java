package neural;

public class MLPRelu implements NeuralNet {

	private Mat w1, w2;
	private Mat b1, b2;
	private double lr = 0.01;

	public MLPRelu(int inputSize, int hiddenSize, int outputSize) {
		w1 = MatUtils.random(hiddenSize, inputSize, -0.1, 0.1);
        b1 = MatUtils.random(hiddenSize, 1, -0.1, 0.1);
        w2 = MatUtils.random(outputSize, hiddenSize, -0.1, 0.1);
        b2 = MatUtils.random(outputSize, 1, -0.1, 0.1);
	}

	public Mat predict(Mat input) {
		Mat z1 = MatUtils.add(MatUtils.dot(w1, input), b1);
		Mat a1 = MatUtils.relu(z1);
		Mat z2 = MatUtils.add(MatUtils.dot(w2, a1), b2);
		Mat a2 = MatUtils.softmax(z2);
		return a2;
	}

	public void train(Mat input, Mat target) {
		Mat z1 = MatUtils.add(MatUtils.dot(w1, input), b1);
		Mat a1 = MatUtils.relu(z1);
		Mat z2 = MatUtils.add(MatUtils.dot(w2, a1), b2);
		Mat a2 = MatUtils.softmax(z2);

		// Backward
		Mat outputGradient = MatUtils.subtract(a2, target); // softmax + cross-entropy simplification
		Mat deltaW2 = MatUtils.dot(outputGradient, MatUtils.transpose(a1));
		MatUtils.scale(deltaW2, lr);
		MatUtils.scale(outputGradient, lr);
		w2 = MatUtils.subtract(w2, deltaW2);
		b2 = MatUtils.subtract(b2, outputGradient);

		Mat hiddenError = MatUtils.dot(MatUtils.transpose(w2), outputGradient);
		Mat hiddenGradient = MatUtils.hadamard(MatUtils.drelu(z1), hiddenError);
		Mat deltaW1 = MatUtils.dot(hiddenGradient, MatUtils.transpose(input));
		MatUtils.scale(deltaW1, lr);
		MatUtils.scale(hiddenGradient, lr);
		w1 = MatUtils.subtract(w1, deltaW1);
		b1 = MatUtils.subtract(b1, hiddenGradient);
	}
}
