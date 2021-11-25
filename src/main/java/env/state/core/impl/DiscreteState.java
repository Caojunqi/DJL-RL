package env.state.core.impl;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import env.state.collector.impl.DiscreteStateCollector;
import env.state.core.IState;
import env.state.space.impl.DiscreteStateSpace;

/**
 * 针对{@link DiscreteStateSpace}型状态空间的State数据
 *
 * @author Caojunqi
 * @date 2021-11-25 15:09
 */
public class DiscreteState implements IState<DiscreteState> {

    private int stateData;

    public DiscreteState(int stateData) {
        this.stateData = stateData;
    }

    @Override
    public DiscreteState clone() {
        return new DiscreteState(stateData);
    }

    @Override
    public Class<DiscreteStateCollector> getCollectorClz() {
        return DiscreteStateCollector.class;
    }

    @Override
    public NDList singleStateList(NDManager manager) {
        return new NDList(manager.create(new int[]{stateData}));
    }

    @Override
    public String toString() {
        return "DiscreteState{" +
                "stateData=" + stateData +
                '}';
    }

    public int getStateData() {
        return stateData;
    }
}
