package neural;

public class Mat {

	public double[][] data;

	public Mat(int rows, int cols) {
		this.data = new double[rows][cols];
	}

	public void init(double val) {
		for (int i = 0; i < rows(); i++)
			for (int j = 0; j < cols(); j++)
				data[i][j] = val;
	}

	public void randomize() {
		for (int i = 0; i < rows(); i++)
			for (int j = 0; j < cols(); j++)
				data[i][j] = Math.random();
	}
	
	public int rows() {return data.length;}
	public int cols() {return data[0].length;}
	
	@Override
	public String toString() {
		var text = new StringBuilder();
		for (int i = 0; i < rows(); i++) {
			text.append("[");
			for (int j = 0; j < cols(); j++) {
				text.append(data[i][j]);
				if (j < cols() - 1)
					text.append(" ");
			}
			text.append("]\n");
		} 
		return text.toString();
	}
}
