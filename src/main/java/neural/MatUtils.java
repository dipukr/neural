package neural;

import java.util.Random;

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
	
	public static Mat relu(Mat m) {
        Mat result = new Mat(m.rows(), m.cols());
        for (int i = 0; i < m.rows(); i++)
            for (int j = 0; j < m.cols(); j++)
                result.data[i][j] = Math.max(0, m.data[i][j]);
        return result;
    }

    public static Mat drelu(Mat m) {
        Mat result = new Mat(m.rows(), m.cols());
        for (int i = 0; i < m.rows(); i++)
            for (int j = 0; j < m.cols(); j++)
                result.data[i][j] = m.data[i][j] > 0 ? 1 : 0;
        return result;
    }

    public static Mat softmax(Mat m) {
        int rows = m.rows(), cols = m.cols();
        Mat result = new Mat(rows, cols);

        for (int j = 0; j < cols; j++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < rows; i++) {
                if (m.data[i][j] > max)
                    max = m.data[i][j];
            }

            double sum = 0.0;
            for (int i = 0; i < rows; i++) {
                result.data[i][j] = Math.exp(m.data[i][j] - max);
                sum += result.data[i][j];
            }

            for (int i = 0; i < rows; i++) {
                result.data[i][j] /= sum;
            }
        }

        return result;
    }
    
    private static final Random rand = new Random(42);

    public static Mat random(int rows, int cols, double min, double max) {
        Mat result = new Mat(rows, cols);
        double range = max - min;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = rand.nextDouble() * range + min;
            }
        }
        return result;
    }
}
