package algorithm;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import env.common.action.Action;
import model.model.BasePolicyModel;
import model.model.BaseValueModel;
import resource.ConstantParameter;
import utils.Memory;

/**
 * 更新模型的算法基类
 *
 * @author Caojunqi
 * @date 2021-10-08 21:04
 */
public abstract class BaseAlgorithm<A extends Action> {

    public abstract void updateModel(NDManager manager, Memory<A> memory, BasePolicyModel<A> policyModel, BaseValueModel valueModel);

    protected NDList estimateAdvantage(NDArray values, NDArray rewards, NDArray masks) {
        NDManager manager = rewards.getManager();
        NDArray deltas = manager.create(rewards.getShape().add(1));
        NDArray advantages = manager.create(rewards.getShape().add(1));

        float prevValue = 0;
        float prevAdvantage = 0;
        for (long i = rewards.getShape().get(0) - 1; i >= 0; i--) {
            NDIndex index = new NDIndex(i);
            int mask = masks.getBoolean(i) ? 0 : 1;
            deltas.set(index, rewards.get(i).add(ConstantParameter.GAMMA * prevValue * mask).sub(values.get(i)));
            advantages.set(index, deltas.get(i).add(ConstantParameter.GAMMA * ConstantParameter.GAE_LAMBDA * prevAdvantage * mask));

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
