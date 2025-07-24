package neural;

public class Mat {

	public double[][] data;

	public Mat(int rows, int cols) {
		this.data = new double[rows][cols];
		for (int i = 0; i < rows(); i++)
			for (int j = 0; j < cols(); j++)
				this.data[i][j] = (2 * Math.random() - 1);
	}

	public Mat(double[][] data) {
		this.data = data;
	}

	public void init(double val) {
		for (int i = 0; i < rows(); i++)
			for (int j = 0; j < cols(); j++)
				data[i][j] = val;
	}
	
	public void random() {
		for (int i = 0; i < rows(); i++)
			for (int j = 0; j < cols(); j++)
				this.data[i][j] = (2 * Math.random() - 1);
	}

	public int rows() {
		return data.length;
	}

	public int cols() {
		return data[0].length;
	}

	public String shape() {
		return String.format("(%d, %d)", rows(), cols());
	}

	@Override
	public String toString() {
		return shape();
//		var text = new StringBuilder();
//		for (int i = 0; i < rows(); i++) {
//			text.append("[");
//			for (int j = 0; j < cols(); j++) {
//				text.append(data[i][j]);
//				if (j < cols() - 1)
//					text.append(" ");
//			}
//			text.append("]\n");
//		}
//		return text.toString();
	}

	public static Mat of(double[][] data) {
		return new Mat(data);
	}

	public static Mat of(double[] data) {
		Mat mat = new Mat(data.length, 1);
		for (int i = 0; i < data.length; i++)
			mat.data[i][0] = data[i];
		return mat;
	}
}
