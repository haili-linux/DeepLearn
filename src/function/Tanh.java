package function;

public class Tanh extends Fuction
{
	public Tanh(){
		super.id = 2;
		super.SourceCode = SourceCodeLib.tanh_code;
		super.name = SourceCodeLib.tanh_code_Name;
		super.SourceCode_derivate = SourceCodeLib.tanh_d_code;
		super.SourceCode_derivate_Name = SourceCodeLib.tanh_d_code_Name;
	}
	@Override
	public double f(double x)
	{
		// TODO: Implement this method
		double x1 = Math.exp(x);
		double x2 = Math.exp(-x);
		return (x1-x2)/(x1+x2);
	}

	@Override
	public double f_derivate(double x1)
	{
		// TODO: Implement this method
		//double x1 = f(x);
		return 1 - x1*x1;
	}


}
