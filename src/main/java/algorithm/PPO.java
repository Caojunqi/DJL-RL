package algorithm;

/**
 * PPO算法核心  Proximal policy optimization algorithms
 *
 * @author Caojunqi
 * @date 2021-09-16 15:44
 */
public class PPO {
    /**
     * 是否渲染环境
     */
    private boolean render = false;
    /**
     * TODO log std for the policy
     */
    private float logStd = -0.0f;
    /**
     * 收益折扣因子
     */
    private float gamma = 0.99f;
    /**
     * GAE计算公式中的λ
     */
    private float gaeLammda = 0.95f;
    /**
     * 进行权重衰减正则化时使用的因子
     */
    private float l2Reg = 1e-3f;
    /**
     * 学习率
     */
    private float learningRate = 3e-4f;
    /**
     * PPO算法中用来限制新旧策略变化范围的ε
     */
    private float clipEpsilon = 0.2f;
    /**
     * 激活的线程数
     */
    private int threadNum = 1;
    /**
     * 随机种子
     */
    private int seed = 1;
    private int minBatchSize = 2048;
    private int evalBatchSize = 2048;
    private int maxIterNum = 500;
    private int logInterval = 1;
    private int saveModelInterval = 0;
    private int gpuIndex = 0;
}
