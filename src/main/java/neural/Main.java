package neural;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Main {
	public static void loadsData(String fileName, List<Mat> dataX, List<Mat> dataY) throws Exception {
		byte[] bytes = Files.readAllBytes(Paths.get(fileName));
		ByteBuffer buffer = ByteBuffer.wrap(bytes);
		buffer.order(ByteOrder.BIG_ENDIAN);

		int count = buffer.getInt();
		int len = 28 * 28;
		
		for (int m = 0; m < count; m++) {
			double[][] data = new double[len][1];
			for (int i = 0; i < len; i++)
				data[i][0] = buffer.getDouble();
			dataX.add(Mat.of(data));
		}
		for (int m = 0; m < count; m++) {
			double[][] data = new double[10][1];
			int val = buffer.getInt();
			data[val][0] = 1.0;
			dataY.add(Mat.of(data));
		}
	}
	public static void loadData(String fileName, List<Mat> dataX, List<Mat> dataY) throws Exception {
		byte[] bytes = Files.readAllBytes(Paths.get(fileName));
		ByteBuffer buffer = ByteBuffer.wrap(bytes);
		buffer.order(ByteOrder.BIG_ENDIAN);

		int count = buffer.getInt();
		int rows = 28;
		int cols = 28;
		
		for (int m = 0; m < count; m++) {
			double[][] data = new double[rows][cols];
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					data[i][j] = buffer.getDouble();
			dataX.add(Mat.of(data));
		}
		for (int m = 0; m < count; m++) {
			double[][] data = new double[10][1];
			int val = buffer.getInt();
			data[val][0] = 1.0;
			dataY.add(Mat.of(data));
		}
	}
	
	public static void main(final String[] args) throws Exception {
		List<Mat> trainX = new ArrayList<>();
		List<Mat> trainY = new ArrayList<>();
		
		List<Mat> testX = new ArrayList<>();
		List<Mat> testY = new ArrayList<>();
		
		loadsData("/home/dkumar/RESEARCH/mnist_train.bin", trainX, trainY);
		loadsData("/home/dkumar/RESEARCH/mnist_test.bin", testX, testY);		
		
		NeuralNet net = new NeuralNet(28 * 28, 64, 10);
		int matchedCount = 0;
		System.out.println("Training....");
		for (int epoch = 0; epoch < 10; epoch++) {
			for (int m = 0; m < trainX.size(); m++)
				net.train(trainX.get(m), trainY.get(m));
			System.out.printf("Epoch %d completed.\n", epoch);
		}
		System.out.println("Testing....");
		for (int m = 0; m < testX.size(); m++) {
			Mat out = net.predict(testX.get(m));
			if (Utils.argmax(out) == Utils.argmax(testY.get(m)))
				matchedCount += 1;
		}
		System.out.printf("Total %d matched out of %d.\n", matchedCount, testX.size());
		System.out.printf("Accuracy rate is %f.\n", ((double)matchedCount/(double)testX.size()) * 100.0);
		
	}
}
