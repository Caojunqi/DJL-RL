package env.common.action.collector;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import env.common.action.Action;
import env.common.action.IActionCollector;
import env.common.action.impl.BoxAction;

/**
 * 连续型动作数据收集器
 *
 * @author Caojunqi
 * @date 2021-09-15 11:45
 */
public class BoxActionCollector implements IActionCollector {

    private double[][] actionData;

    public BoxActionCollector(int batchSize) {
        this.actionData = new double[batchSize][];
    }

    @Override
    public void addAction(int index, Action action) {
        this.actionData[index] = ((BoxAction) action).getActionData();
    }

    @Override
    public NDArray createNDArray(NDManager manager) {
        return manager.create(actionData);
    }
}
