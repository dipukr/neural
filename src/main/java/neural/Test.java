package neural;

public class Test {
	public static void main(final String[] args) throws Exception {
		long start = System.currentTimeMillis();
		Mat m = MatUtils.random(10000, 10000, -1, 1);
		Mat n = MatUtils.random(10000, 10000, -1, 1);
		Mat dot = MatUtils.dot(m, n);
		
		System.out.println(System.currentTimeMillis() - start);
	}
}
