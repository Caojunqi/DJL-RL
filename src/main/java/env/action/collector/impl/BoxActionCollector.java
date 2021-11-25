package env.action.collector.impl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import env.action.collector.IActionCollector;
import env.action.core.IAction;
import env.action.core.impl.BoxAction;

/**
 * 连续型动作数据收集器
 *
 * @author Caojunqi
 * @date 2021-09-15 11:45
 */
public class BoxActionCollector implements IActionCollector {

    private float[][] actionDatas;

    public BoxActionCollector(int batchSize) {
        this.actionDatas = new float[batchSize][];
    }

    @Override
    public void addAction(int index, IAction action) {
        this.actionDatas[index] = ((BoxAction) action).getActionData();
    }

    @Override
    public NDArray createNDArray(NDManager manager) {
        return manager.create(actionDatas);
    }
}
