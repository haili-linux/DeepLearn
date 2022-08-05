package DeltaOptimizer;

public class Momentum implements BaseOptimizerInterface {

    int index;
    int len;
    double nl;
    double dnl;
    double[] m;

    public Momentum(int delta_number , double nl ) {
        index = 0;
        m = new double[delta_number];
        len = m.length - 1;
        this.nl = nl;
        dnl = 1 - nl;
    }

    @Override
    public void init() {
        m = new double[m.length];
        index = 0;
    }

    @Override
    public double DELTA(double delta) {

        double r = m[index] =  nl * m[index] + dnl * delta;

        if(index == len)
            index = 0;
        else
            index++;

        return r;
    }
}
