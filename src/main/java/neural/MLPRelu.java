package neural;

public class MLPRelu implements NeuralNet {

	private Mat w1, w2;
	private Mat b1, b2;
	private double lr = 0.001;

	public MLPRelu(int inputSize, int hiddenSize, int outputSize) {
		w1 = Mats.random(hiddenSize, inputSize, -0.1, 0.1);
        b1 = Mats.random(hiddenSize, 1, -0.1, 0.1);
        w2 = Mats.random(outputSize, hiddenSize, -0.1, 0.1);
        b2 = Mats.random(outputSize, 1, -0.1, 0.1);
	}

	public Mat predict(Mat input) {
		Mat z1 = Mats.add(Mats.dot(w1, input), b1);
		Mat a1 = Mats.relu(z1);
		Mat z2 = Mats.add(Mats.dot(w2, a1), b2);
		Mat a2 = Mats.softmax(z2);
		return a2;
	}

	public void train(Mat input, Mat target) {
		Mat z1 = Mats.add(Mats.dot(w1, input), b1);
		Mat a1 = Mats.relu(z1);
		Mat z2 = Mats.add(Mats.dot(w2, a1), b2);
		Mat a2 = Mats.softmax(z2);

		// Backward
		Mat outputGradient = Mats.subtract(a2, target); // softmax + cross-entropy simplification
		Mat deltaW2 = Mats.dot(outputGradient, Mats.transpose(a1));
		Mat hiddenError = Mats.dot(Mats.transpose(w2), outputGradient);
		Mat hiddenGradient = Mats.hadamard(Mats.drelu(z1), hiddenError);
		Mat deltaW1 = Mats.dot(hiddenGradient, Mats.transpose(input));
		
		Mats.scale(deltaW2, lr);
		Mats.scale(outputGradient, lr);
		Mats.scale(deltaW1, lr);
		Mats.scale(hiddenGradient, lr);

		w2 = Mats.subtract(w2, deltaW2);
		b2 = Mats.subtract(b2, outputGradient);
		w1 = Mats.subtract(w1, deltaW1);
		b1 = Mats.subtract(b1, hiddenGradient);
	}
}
