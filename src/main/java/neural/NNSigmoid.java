package neural;

import java.util.ArrayList;
import java.util.List;

public class NNSigmoid implements NeuralNet {

	private Mat[] weights;
	private Mat[] biases;
	private double lr = 0.01;

	public NNSigmoid(int[] topology) {
		this.weights = new Mat[topology.length - 1];
		this.biases = new Mat[topology.length - 1];
		
		for (int i = 1; i < topology.length; i++) {
			this.weights[i - 1] = MatUtils.random(topology[i], topology[i - 1], -1, 1);
			this.biases[i - 1] = MatUtils.random(topology[i], 1, -1, 1);
		}
	}

	public Mat predict(Mat input) {
		Mat a = input;
		for (int i = 0; i < weights.length; i++) {
			Mat z = MatUtils.add(MatUtils.dot(weights[i], a), biases[i]);
			a = MatUtils.sigmoid(z);
		}
		return a;
	}

	public void train(Mat input, Mat target) {
		Mat[] as = new Mat[weights.length + 1];
		Mat[] zs = new Mat[weights.length];
		
		// Forward pass
		Mat a = input;
		as[0] = a;
		for (int i = 0; i < weights.length; i++) {
			Mat z = MatUtils.add(MatUtils.dot(weights[i], a), biases[i]);
			zs[i] = z;
			a = MatUtils.sigmoid(z);
			as[i + 1] = a;
		}

		// Backward pass
		List<Mat> delta = new ArrayList<>();
		int last = weights.length - 1;

		Mat outputError = MatUtils.subtract(as[last + 1], target);
		Mat outputGradient = MatUtils.hadamard(outputError, MatUtils.dsigmoid(as[last + 1]));
		delta.add(0, outputGradient);

		for (int l = last - 1; l >= 0; l--) {
			Mat wNextT = MatUtils.transpose(weights[l + 1]);
			Mat d = MatUtils.hadamard(MatUtils.dot(wNextT, delta.get(0)), MatUtils.dsigmoid(as[l + 1]));
			delta.add(0, d);
		}

		// Update weights and biases
		for (int i = 0; i < weights.length; i++) {
			Mat dw = MatUtils.dot(delta.get(i), MatUtils.transpose(as[i]));
			MatUtils.scale(dw, lr);
			MatUtils.scale(delta.get(i), lr);
			weights[i] = MatUtils.subtract(weights[i], dw);
			biases[i] = MatUtils.subtract(biases[i], delta.get(i));
		}
	}
}
