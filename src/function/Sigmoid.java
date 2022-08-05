package function;

import jcuda.driver.*;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.nvrtc.JNvrtc.*;
import static jcuda.nvrtc.JNvrtc.nvrtcDestroyProgram;

public class Sigmoid extends Fuction
{
	public Sigmoid(){
		super.id = 1;
		super.SourceCode = SourceCodeLib.sigmoid_code;
		super.name = SourceCodeLib.sigmoid_code_Name;
		super.SourceCode_derivate = SourceCodeLib.sigmoid_d_code;
		super.SourceCode_derivate_Name = SourceCodeLib.sigmoid_d_code_Name;
	}
	
	@Override
	public double f(double x)
	{
		// TODO: Implement this method
		return 1.0 / (1.0 + Math.exp(-x));
	}

	@Override
	public double f_derivate(double x1)
	{
		// TODO: Implement this method
		//double x1 = f(x);
		return x1 * (1.0 - x1);
	}
}
