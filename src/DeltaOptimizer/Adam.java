package DeltaOptimizer;

public class Adam implements BaseOptimizerInterface {

    double v1;
    double v2;
    double e;
    double dv1;
    double dv2;
    double v1_t;
    double v2_t;
    double[] m;
    double[] v;
    int len;
    int index;

    public Adam(int delta_number, double v1, double v2, double e){
        m = new double[delta_number];
        v = new double[delta_number];
        len = m.length -1;
        this.v1 = v1;
        this.v2 = v2;
        this.e = e;
        dv1 = 1 - v1;
        dv2 = 1 - v2;
        v1_t = 1;
        v2_t = 1;
        index = 0;
    }

    @Override
    public void init() {
        v1_t = 1;
        v2_t = 1;
        index = 0;
        m = new double[m.length];
        v = new double[v.length];
    }

    @Override
    public double DELTA(double delta) {

        m[index] = v1 * m[index] + dv1 * delta;
        v[index] = v2 * v[index] + dv2 * delta * delta;

        v1_t *= v1;
        v2_t *= v2;

        double m_ = m[index] / (1 - v1_t);
        double v_ = v[index] / (1 - v2_t);

        if(index == len)
            index = 0;
        else
            index++;

        return   m_ / ( Math.sqrt(v_) + e );
    }
}
