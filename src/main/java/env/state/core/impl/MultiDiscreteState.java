package env.state.core.impl;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import env.state.collector.impl.MultiDiscreteStateCollector;
import env.state.core.IState;
import env.state.space.impl.MultiDiscreteStateSpace;

import java.util.Arrays;

/**
 * 针对{@link MultiDiscreteStateSpace}型状态空间的State数据
 *
 * @author Caojunqi
 * @date 2021-11-25 15:17
 */
public class MultiDiscreteState implements IState<MultiDiscreteState> {

    private int[] stateData;

    public MultiDiscreteState(int[] stateData) {
        this.stateData = stateData;
    }

    @Override
    public MultiDiscreteState clone() {
        return new MultiDiscreteState(stateData.clone());
    }

    @Override
    public Class<MultiDiscreteStateCollector> getCollectorClz() {
        return MultiDiscreteStateCollector.class;
    }

    @Override
    public NDList singleStateList(NDManager manager) {
        return new NDList(manager.create(new int[][]{stateData}));
    }

    @Override
    public String toString() {
        return "MultiDiscreteState{" +
                "stateData=" + Arrays.toString(stateData) +
                '}';
    }

    public int[] getStateData() {
        return stateData;
    }
}
