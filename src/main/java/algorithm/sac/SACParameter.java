package algorithm.sac;

/**
 * SAC算法参数常量类
 *
 * @author Caojunqi
 * @date 2021-10-12 17:17
 */
public final class SACParameter {
    /**
     * 神经网络隐藏层参数个数
     */
    public final static int[] NETS_HIDDEN_SIZES = new int[]{64, 64};
    /**
     * 策略模型优化学习率
     */
    public final static float POLICY_LR = 3e-4f;
    /**
     * Q函数模型优化学习率
     */
    public final static float QF_LR = 3e-4f;
    /**
     * 策略模型优化权重衰减系数
     */
    public final static float POLICY_WEIGHT_DECAY = 1.e-5f;
    /**
     * Q函数模型优化权重衰减系数
     */
    public final static float Q_WEIGHT_DECAY = 1.e-5f;
    public final static float ENTROPY_SCALE = 1.f;
    public final static boolean AUTO_ALPHA = true;
    public final static float MAX_ALPHA = 10;
    public final static float MIN_ALPHA = 0.01f;

}
