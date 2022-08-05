package function;

public class LRelu extends Fuction{
    public  LRelu(){
        super.id = 4;
        super.SourceCode = SourceCodeLib.Lrelu_code;
        super.name = SourceCodeLib.Lrelu_code_Name;
        super.SourceCode_derivate = SourceCodeLib.lrule_d_code;
        super.SourceCode_derivate_Name = SourceCodeLib.lrule_d_code_Name;
    }
    @Override
    public double f(double x) {
        return (x>0) ? x : 0.001*x;
    }

    @Override
    public double f_derivate(double x) {
        return (x>0) ? 1 : 0.001;
    }
}
