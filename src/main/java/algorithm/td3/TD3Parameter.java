package algorithm.td3;

/**
 * TD3算法参数常量类
 *
 * @author Caojunqi
 * @date 2021-11-01 11:45
 */
public final class TD3Parameter {
    /**
     * 神经网络隐藏层参数个数
     */
    public final static int[] NETS_HIDDEN_SIZES = new int[]{20, 20};
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
     * 动作扰乱因子方差
     */
    public final static float ACTION_NOISE_STD = 0.2f;
    /**
     * 动作扰乱因子范围
     */
    public final static float ACTION_NOISE_CLIPPING_RANGE = 0.5f;
}
