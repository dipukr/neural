package neural;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Test {
	public static void main(final String[] args) throws Exception {
		long start = System.currentTimeMillis();
		INDArray A = Nd4j.rand(10000, 10000);
		INDArray B = Nd4j.rand(10000, 10000);

		// Time the matrix multiplication
		INDArray C = A.mmul(B);

		System.out.println(System.currentTimeMillis() - start);
	}
}
