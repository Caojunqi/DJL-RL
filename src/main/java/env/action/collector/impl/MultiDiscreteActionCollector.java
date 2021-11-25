package env.action.collector.impl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import env.action.collector.IActionCollector;
import env.action.core.IAction;
import env.action.core.impl.MultiDiscreteAction;

/**
 * 多维离散型动作数据收集器
 *
 * @author Caojunqi
 * @date 2021-09-15 15:54
 */
public class MultiDiscreteActionCollector implements IActionCollector {

    private int[][] actionDatas;

    public MultiDiscreteActionCollector(int batchSize) {
        this.actionDatas = new int[batchSize][];
    }

    @Override
    public void addAction(int index, IAction action) {
        this.actionDatas[index] = ((MultiDiscreteAction) action).getActionData();
    }

    @Override
    public NDArray createNDArray(NDManager manager) {
        return manager.create(actionDatas);
    }
}
