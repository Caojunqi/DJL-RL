package algorithm.ppo;

/**
 * PPO算法参数常量类
 *
 * @author Caojunqi
 * @date 2021-10-12 14:56
 */
public final class PPOParameter {
    /**
     * 策略模型隐藏层参数个数
     */
    public final static int[] POLICY_MODEL_HIDDEN_SIZE = new int[]{128, 128};
    /**
     * 状态价值函数近似模型隐藏层参数个数
     */
    public final static int[] CRITIC_MODEL_HIDDEN_SIZE = new int[]{128, 128};
    /**
     * PPO算法中用来限制新旧策略变化范围的ε
     */
    public final static float CLIP_EPSILON = 0.2f;
    public final static float RATIO_LOWER_BOUND = 1.0f - CLIP_EPSILON;
    public final static float RATIO_UPPER_BOUND = 1.0f + CLIP_EPSILON;
}
