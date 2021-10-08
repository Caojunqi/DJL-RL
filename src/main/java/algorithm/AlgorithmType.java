package algorithm;

import env.common.action.Action;

/**
 * 算法类型
 *
 * @author Caojunqi
 * @date 2021-10-08 21:26
 */
public enum AlgorithmType {

    /**
     * PPO算法，针对离散型动作空间
     */
    PPO_DISCRETE(1) {
        @Override
        public BaseAlgorithm<? extends Action> getAlgorithm() {
            return PPODiscrete.INSTANCE;
        }
    },
    /**
     * PPO算法，针对连续型动作空间
     */
    PPO_CONTINUOUS(2) {
        @Override
        public BaseAlgorithm<? extends Action> getAlgorithm() {
            return PPOContinuous.INSTANCE;
        }
    },
    ;

    /**
     * 编码
     */
    private int code;

    AlgorithmType(int code) {
        this.code = code;
    }

    public abstract <A extends Action> BaseAlgorithm<A> getAlgorithm();

}
