package resource;

/**
 * 参数常量类
 *
 * @author Caojunqi
 * @date 2021-09-30 15:15
 */
public final class ConstantParameter {
    /**
     * 是否渲染环境
     */
    public final static boolean RENDER = false;
    /**
     * 在进行连续型动作选择时，对动作方差进行的微调
     */
    public final static float LOG_STD = -0.0f;
    /**
     * 收益折扣因子
     */
    public final static float GAMMA = 0.99f;
    /**
     * GAE计算公式中的λ
     */
    public final static float GAE_LAMBDA = 0.95f;
    /**
     * 进行权重衰减正则化时使用的因子
     */
    public final static float L2_REG = 1e-3f;
    /**
     * 学习率
     */
    public final static float LEARNING_RATE = 3e-4f;
    /**
     * 激活的线程数
     */
    public final static int THREAD_NUM = 1;
    /**
     * 随机种子
     */
    public final static int SEED = 1;
    /**
     * 每次进行模型更新最少需要收集的样本个数
     */
    public final static int MIN_BATCH_SIZE = 2048;
    /**
     * 在测试模型性能时最少需要收集的样本个数
     */
    public final static int EVAL_BATCH_SIZE = 2048;
    /**
     * 模型最大更新次数
     */
    public final static int MAX_ITER_NUM = 500;
    /**
     * 日志输出间隔
     */
    public final static int LOG_INTERVAL = 1;
    /**
     * 模型保存间隔
     */
    public final static int SAVE_MODEL_INTERVAL = 0;
    /**
     * 可用GPU索引
     */
    public final static int GPU_INDEX = 0;
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
}
