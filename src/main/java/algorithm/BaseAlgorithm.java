package algorithm;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import env.action.core.IAction;
import env.state.core.IState;
import utils.Memory;

/**
 * 更新模型的算法基类
 *
 * @author Caojunqi
 * @date 2021-10-08 21:04
 */
public abstract class BaseAlgorithm<S extends IState<S>, A extends IAction> {
    /**
     * NDArray管理类
     */
    protected NDManager manager;
    /**
     * 样本缓存
     */
    protected Memory<S, A> memory = new Memory<>();

    public BaseAlgorithm(NDManager manager) {
        this.manager = manager;
    }

    /**
     * 根据指定环境状态选择一个合适的动作
     *
     * @param state 环境当前状态
     * @return action 接下来应采取的动作
     */
    public abstract A selectAction(S state);

    /**
     * 采用贪婪策略，为指定环境状态选择一个确定的动作
     *
     * @param state 环境当前状态
     * @return 确定动作
     */
    public abstract A greedyAction(S state);

    public abstract void updateModel();

    /**
     * 重置样本缓存
     */
    public void resetMemory() {
        this.memory.reset();
    }

    public void collect(S state, A action, boolean done, S nextState, float reward) {
        memory.addTransition(state, action, done, nextState, reward);
    }

    protected NDList estimateAdvantage(NDArray values, NDArray rewards, NDArray masks) {
        NDManager manager = rewards.getManager();
        NDArray deltas = manager.create(rewards.getShape());
        NDArray advantages = manager.create(rewards.getShape());

        float prevValue = 0;
        float prevAdvantage = 0;
        for (long i = rewards.getShape().get(0) - 1; i >= 0; i--) {
            NDIndex index = new NDIndex(i);
            int mask = masks.getBoolean(i) ? 0 : 1;
            deltas.set(index, rewards.get(i).add(CommonParameter.GAMMA * prevValue * mask).sub(values.get(i)));
            advantages.set(index, deltas.get(i).add(CommonParameter.GAMMA * CommonParameter.GAE_LAMBDA * prevAdvantage * mask));

            prevValue = values.getFloat(i);
            prevAdvantage = advantages.getFloat(i);
        }

        NDArray expected_returns = values.add(advantages);
        NDArray advantagesMean = advantages.mean();
        NDArray advantagesStd = advantages.sub(advantagesMean).pow(2).sum().div(advantages.size() - 1).sqrt();
        advantages = advantages.sub(advantagesMean).div(advantagesStd);

        return new NDList(expected_returns, advantages);
    }

    protected NDArray getSample(NDManager subManager, NDArray array, int[] index) {
        Shape shape = Shape.update(array.getShape(), 0, index.length);
        NDArray sample = subManager.zeros(shape, array.getDataType());
        for (int i = 0; i < index.length; i++) {
            sample.set(new NDIndex(i), array.get(index[i]));
        }
        return sample;
    }
}
