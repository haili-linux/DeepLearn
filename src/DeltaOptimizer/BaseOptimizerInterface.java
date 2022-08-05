package DeltaOptimizer;


//梯度下降优化器
public interface BaseOptimizerInterface {

    /**
     * 重置优化器
     */
    public void init();

    /**
     * 梯度优化
     * @param delta 输入梯度
     * @return 优化后的梯度
     */
    public double DELTA(double delta);

}
