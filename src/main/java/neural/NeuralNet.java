package neural;

public class NeuralNet {

	private Mat w1, w2;
	private Mat b1, b2;
	private double lr = 0.00001;

	public NeuralNet(int inputSize, int hiddenSize, int outputSize) {
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
		// Forward
		Mat z1 = MatUtils.add(MatUtils.dot(w1, input), b1);
		Mat a1 = MatUtils.relu(z1);
		Mat z2 = MatUtils.add(MatUtils.dot(w2, a1), b2);
		Mat a2 = MatUtils.relu(z2);

		// Backward
		Mat outputError = MatUtils.subtract(target, a2);
		Mat outputGradient = MatUtils.hadamard(MatUtils.drelu(a2), outputError);
		Mat hiddenT = MatUtils.transpose(a1);
		Mat deltaW2 = MatUtils.dot(outputGradient, hiddenT);
		MatUtils.scale(deltaW2, lr);
		MatUtils.scale(outputGradient, lr);

		w2 = MatUtils.add(w2, deltaW2);
		b2 = MatUtils.add(b2, outputGradient);

		Mat w2T = MatUtils.transpose(w2);
		Mat hiddenError = MatUtils.dot(w2T, outputError);
		Mat hiddenGradient = MatUtils.hadamard(MatUtils.drelu(a1), hiddenError);
		Mat inputT = MatUtils.transpose(input);
		Mat deltaW1 = MatUtils.dot(hiddenGradient, inputT);
		MatUtils.scale(deltaW1, lr);
		MatUtils.scale(hiddenGradient, lr);

		w1 = MatUtils.add(w1, deltaW1);
		b1 = MatUtils.add(b1, hiddenGradient);
	}
}
