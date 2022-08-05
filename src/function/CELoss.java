package function;

public class CELoss extends Fuction{
    //交叉熵
    public CELoss(){
        super.id = 11;
    }

    @Override
    public double f(double y, double t) {
        return -(t*Math.log10(y) + (1-t)*Math.log10(1-y));
    }

    @Override
    public double f_derivate(double y, double t) {
        return (y-t)/(y*(1-y));
    }

}
