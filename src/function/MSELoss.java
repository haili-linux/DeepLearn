package function;

//回归损伤函数
public class MSELoss extends Fuction
{

	public MSELoss(){
		super.id = 10;
	}

	@Override
	public double f(double y, double t)
	{
		// TODO: Implement this method
		double x = t - y;
		return x*x/0.5;
	}

	@Override
	public double f_derivate(double y, double t)
	{
		// TODO: Implement this method
		return -(t - y);
	}
}
