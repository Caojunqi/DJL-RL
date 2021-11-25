package env.state.collector.impl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import env.state.collector.IStateCollector;
import env.state.core.IState;
import env.state.core.impl.MultiDiscreteState;

/**
 * 多维离散型状态数据收集器
 *
 * @author Caojunqi
 * @date 2021-11-25 15:18
 */
public class MultiDiscreteStateCollector implements IStateCollector {

    private int[][] stateDatas;

    public MultiDiscreteStateCollector(int batchSize) {
        this.stateDatas = new int[batchSize][];
    }

    @Override
    public void addState(int index, IState state) {
        this.stateDatas[index] = ((MultiDiscreteState) state).getStateData();
    }

    @Override
    public NDArray createNDArray(NDManager manager) {
        return manager.create(stateDatas);
    }
}
