package function;

public class CESLoss extends Fuction{
    public CESLoss(){
        super.id = 12;
    }

    @Override
    public double f(double y, double t) {
        return -t * Math.log10(y);
    }

    @Override
    public double f_derivate(double y, double t) {
           return -t/y;
    }
}
