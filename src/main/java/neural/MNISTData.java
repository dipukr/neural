package neural;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class MNISTData {
	
	private List<Mat> trainX = new ArrayList<>();
	private List<Mat> trainY = new ArrayList<>();
	private List<Mat> testX = new ArrayList<>();
	private List<Mat> testY = new ArrayList<>();
	
	public MNISTData(String fileName) throws Exception {
		byte[] bytes = Files.readAllBytes(Paths.get(fileName));
		ByteBuffer buffer = ByteBuffer.wrap(bytes);
		buffer.order(ByteOrder.BIG_ENDIAN);

		final int trainCount = 60_000;
		final int testCount = 10_000;
		final int len = 28 * 28;
		
		for (int m = 0; m < trainCount; m++) {
			double[][] data = new double[len][1];
			for (int i = 0; i < len; i++)
				data[i][0] = buffer.getDouble();
			trainX.add(Mat.of(data));
		}
		for (int m = 0; m < testCount; m++) {
			double[][] data = new double[len][1];
			for (int i = 0; i < len; i++)
				data[i][0] = buffer.getDouble();
			testX.add(Mat.of(data));
		}
		for (int m = 0; m < trainCount; m++) {
			double[][] data = new double[10][1];
			int val = buffer.getInt();
			data[val][0] = 1.0;
			trainY.add(Mat.of(data));
		}
		for (int m = 0; m < testCount; m++) {
			double[][] data = new double[10][1];
			int val = buffer.getInt();
			data[val][0] = 1.0;
			testY.add(Mat.of(data));
		}
	}

	public List<Mat> getTrainX() {return trainX;}
	public List<Mat> getTrainY() {return trainY;}
	public List<Mat> getTestX() {return testX;}
	public List<Mat> getTestY() {return testY;}
}
