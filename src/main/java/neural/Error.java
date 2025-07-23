package neural;

public class Error {
	
	public static final String WRONG_INPUT = "invalid input value";
	
	public static void fatal(String messge) {
		System.out.println("Fatal error: " + messge);
		System.exit(1);
	}
}
