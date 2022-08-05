package function;

public class Relu extends Fuction
{
	public Relu(){
		super.id = 3;
		super.SourceCode = SourceCodeLib.relu_code;
		super.name = SourceCodeLib.relu_code_Name;
		super.SourceCode_derivate = SourceCodeLib.rule_d_code;
		super.SourceCode_derivate_Name = SourceCodeLib.rule_d_code_Name;
	}
	@Override
	public double f(double x)
	{
		// TODO: Implement this method
		return (x>0) ? x:0;
	}

	@Override
	public double f_derivate(double x)
	{
		// TODO: Implement this method
		return (x>0) ? 1:0;
	}


}

