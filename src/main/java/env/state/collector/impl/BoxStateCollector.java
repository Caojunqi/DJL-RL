package env.state.collector.impl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import env.state.collector.IStateCollector;
import env.state.core.IState;
import env.state.core.impl.BoxState;

/**
 * 连续型状态数据收集器
 *
 * @author Caojunqi
 * @date 2021-11-25 15:02
 */
public class BoxStateCollector implements IStateCollector {

    private float[][] stateDatas;

    public BoxStateCollector(int batchSize) {
        this.stateDatas = new float[batchSize][];
    }

    @Override
    public void addState(int index, IState state) {
        this.stateDatas[index] = ((BoxState) state).getStateData();
    }

    @Override
    public NDArray createNDArray(NDManager manager) {
        return manager.create(stateDatas);
    }
}
