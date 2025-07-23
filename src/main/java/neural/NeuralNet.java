package neural;

public class NeuralNet {

	private Mat w1, w2;
	private Mat b1, b2;
	private double lr = 0.1;

	public NeuralNet(int inputSize, int hiddenSize, int outputSize) {
		this.w1 = new Mat(hiddenSize, inputSize);
		this.w2 = new Mat(outputSize, hiddenSize);
		this.b1 = new Mat(hiddenSize, 1);
		this.b2 = new Mat(outputSize, 1);
		this.w1.randomize();
		this.w2.randomize();
		this.b1.randomize();
		this.b2.randomize();
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

		// Backward
		Mat outputError = MatUtils.subtract(target, a2);
		Mat outputGradient = MatUtils.hadamard(MatUtils.dsigmoid(a2), outputError);
		Mat hiddenT = MatUtils.transpose(a1);
		Mat deltaW2 = MatUtils.dot(outputGradient, hiddenT);
		MatUtils.scale(deltaW2, lr);
		MatUtils.scale(outputGradient, lr);

		w2 = MatUtils.add(w2, deltaW2);
		b2 = MatUtils.add(b2, outputGradient);

		Mat w2T = MatUtils.transpose(w2);
		Mat hiddenError = MatUtils.dot(w2T, outputError);
		Mat hiddenGradient = MatUtils.hadamard(MatUtils.dsigmoid(a1), hiddenError);
		Mat inputT = MatUtils.transpose(input);
		Mat deltaW1 = MatUtils.dot(hiddenGradient, inputT);
		MatUtils.scale(deltaW1, lr);
		MatUtils.scale(hiddenGradient, lr);

		w1 = MatUtils.add(w1, deltaW1);
		b1 = MatUtils.add(b1, hiddenGradient);
	}
}
