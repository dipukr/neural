package neural;

import java.util.Random;
import java.util.function.Function;

public class MatUtils {

	private static final Random rand = new Random(42);

	public static void verifyDim(Mat m, Mat n) {
		if (m.rows() != n.rows() || m.cols() != n.cols())
			Error.fatal("Operation can not be performed.");
	}

	public static Mat add(Mat m, Mat n) {
		verifyDim(m, n);
		Mat r = new Mat(m.rows(), m.cols());
		for (int i = 0; i < m.rows(); i++)
			for (int j = 0; j < m.cols(); j++)
				r.data[i][j] = m.data[i][j] + n.data[i][j];
		return r;
	}

	public static Mat subtract(Mat m, Mat n) {
		verifyDim(m, n);
		Mat r = new Mat(m.rows(), m.cols());
		for (int i = 0; i < m.rows(); i++)
			for (int j = 0; j < m.cols(); j++)
				r.data[i][j] = m.data[i][j] - n.data[i][j];
		return r;
	}

	public static Mat hadamard(Mat m, Mat n) {
		verifyDim(m, n);
		Mat r = new Mat(m.rows(), m.cols());
		for (int i = 0; i < m.rows(); i++)
			for (int j = 0; j < m.cols(); j++)
				r.data[i][j] = m.data[i][j] * n.data[i][j];
		return r;
	}

	public static Mat dot(Mat m, Mat n) {
		if (m.cols() != n.rows())
			Error.fatal("Operation can not be performed.");
		Mat r = new Mat(m.rows(), n.cols());
		for (int i = 0; i < m.rows(); i++) {
			for (int j = 0; j < n.cols(); j++) {
				double sum = 0;
				for (int k = 0; k < m.cols(); k++)
					sum += m.data[i][k] * n.data[k][j];
				r.data[i][j] = sum;
			}
		}
		return r;
	}

	public static Mat multiply(Mat m, Mat n) {
		if (m.cols() != n.rows())
			Error.fatal("Operation can not be performed.");
		Mat r = new Mat(m.rows(), n.cols());
		for (int i = 0; i < m.rows(); i++) {
			for (int j = 0; j < n.cols(); j++) {
				double sum = 0;
				for (int k = 0; k < m.cols(); k++)
					sum += m.data[i][k] * n.data[k][j];
				r.data[i][j] = sum;
			}
		}
		return r;
	}

	public static Mat transpose(Mat m) {
		Mat r = new Mat(m.cols(), m.rows());
		for (int i = 0; i < m.rows(); i++)
			for (int j = 0; j < m.cols(); j++)
				r.data[j][i] = m.data[i][j];
		return r;
	}

	public static void scale(Mat m, double scalar) {
		for (int i = 0; i < m.rows(); i++)
			for (int j = 0; j < m.cols(); j++)
				m.data[i][j] *= scalar;
	}

	public static Mat sigmoid(Mat m) {
		Mat r = new Mat(m.rows(), m.cols());
		for (int i = 0; i < m.rows(); i++)
			for (int j = 0; j < m.cols(); j++)
				r.data[i][j] = Utils.sigmoid(m.data[i][j]);
		return r;
	}

	public static Mat dsigmoid(Mat m) {
		Mat r = new Mat(m.rows(), m.cols());
		for (int i = 0; i < m.rows(); i++)
			for (int j = 0; j < m.cols(); j++)
				r.data[i][j] = Utils.dsigmoid(m.data[i][j]);
		return r;
	}

	public static Mat relu(Mat m) {
		Mat r = new Mat(m.rows(), m.cols());
		for (int i = 0; i < m.rows(); i++)
			for (int j = 0; j < m.cols(); j++)
				r.data[i][j] = Utils.relu(m.data[i][j]);
		return r;
	}

	public static Mat drelu(Mat m) {
		Mat r = new Mat(m.rows(), m.cols());
		for (int i = 0; i < m.rows(); i++)
			for (int j = 0; j < m.cols(); j++)
				r.data[i][j] = Utils.drelu(m.data[i][j]);
		;
		return r;
	}

	public static Mat softmax(Mat m) {
		Mat r = new Mat(m.rows(), m.cols());
		for (int j = 0; j < m.cols(); j++) {
			double max = Double.NEGATIVE_INFINITY;
			for (int i = 0; i < m.rows(); i++) {
				if (m.data[i][j] > max)
					max = m.data[i][j];
			}
			double sum = 0.0;
			for (int i = 0; i < m.rows(); i++) {
				r.data[i][j] = Math.exp(m.data[i][j] - max);
				sum += r.data[i][j];
			}
			for (int i = 0; i < m.rows(); i++)
				r.data[i][j] /= sum;
		}
		return r;
	}

	public static Mat random(int rows, int cols, double min, double max) {
		Mat r = new Mat(rows, cols);
		double range = max - min;
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				r.data[i][j] = rand.nextDouble() * range + min;
		return r;
	}

	public static Mat map(Mat m, Function<Double, Double> fn) {
		Mat r = new Mat(m.rows(), m.cols());
		for (int i = 0; i < m.rows(); i++)
			for (int j = 0; j < m.cols(); j++)
				r.data[i][j] = fn.apply(m.data[i][j]);
		return r;
	}

	public static void mapInPlace(Mat m, Function<Double, Double> fn) {
		for (int i = 0; i < m.rows(); i++)
			for (int j = 0; j < m.cols(); j++)
				m.data[i][j] = fn.apply(m.data[i][j]);
	}
}
