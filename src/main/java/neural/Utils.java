package neural;

import java.util.Random;

public class Utils {
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
}
