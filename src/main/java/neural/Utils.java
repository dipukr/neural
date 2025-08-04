package neural;

public class Utils {
	public static double sigmoid(double val) {
		return 1.0 / (1.0 + Math.exp(-val));
	}
	
	public static double dsigmoid(double val) {
		val = sigmoid(val);
		return val * (1 - val);
	}
	
	public static double relu(double val) {
		return Math.max(0, val);
	}
	
	public static double drelu(double val) {
		return Math.max(0, val);
	}
	
	public static int argmax(Mat mat) {
		if (mat == null || mat.data.length == 0 || mat.data[0].length != 1)
			Error.fatal("Input must be a non-empty column matrix.");
		int maxIndex = 0;
		double maxValue = mat.data[0][0];
		for (int i = 1; i < mat.data.length; i++) {
			if (mat.data[i][0] > maxValue) {
				maxValue = mat.data[i][0];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	public static int argmax(float[] data) {
		int maxIndex = 0;
		float maxValue = data[0];
		for (int i = 1; i < data.length; i++) {
			if (data[i] > maxValue) {
				maxValue = data[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	public static float[] softmax(float[] input) {
		float max = Float.NEGATIVE_INFINITY;
		for (float v : input)
			max = Math.max(max, v);
		float sum = 0;
		for (int i = 0; i < input.length; i++) {
			input[i] = (float) Math.exp(input[i] - max);
			sum += input[i];
		}
		for (int i = 0; i < input.length; i++)
			input[i] /= sum;
		return input;
	}
}
