package env.common.action.collector;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import env.common.action.Action;
import env.common.action.IActionCollector;
import env.common.action.impl.DiscreteAction;

/**
 * 离散型动作数据收集器
 *
 * @author Caojunqi
 * @date 2021-09-15 15:54
 */
public class DiscreteActionCollector implements IActionCollector {

    private int[] actionData;

    public DiscreteActionCollector(int batchSize) {
        this.actionData = new int[batchSize];
    }

    @Override
    public void addAction(int index, Action action) {
        this.actionData[index] = ((DiscreteAction) action).getActionData();
    }

    @Override
    public NDArray createNDArray(NDManager manager) {
        return manager.create(actionData);
    }
}
