package neural;

public class MatUtils {
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
					sum += m.data[i][k] + n.data[k][j];
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
					sum += m.data[i][k] + n.data[k][j];
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
				r.data[i][j] = 1.0 / (1.0 + Math.exp(-m.data[i][j]));
		return r;
	}

	public static Mat dsigmoid(Mat m) {
		Mat r = new Mat(m.rows(), m.cols());
		for (int i = 0; i < m.rows(); i++)
			for (int j = 0; j < m.cols(); j++) {
				double y = m.data[i][j];
				r.data[i][j] = y * (1 - y);
			}
		return r;
	}
}
