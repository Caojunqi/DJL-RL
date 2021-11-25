package env.state.core.impl;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import env.state.collector.impl.BoxStateCollector;
import env.state.core.IState;
import env.state.space.impl.BoxStateSpace;

import java.util.Arrays;

/**
 * 针对{@link BoxStateSpace}型状态空间的State数据
 *
 * @author Caojunqi
 * @date 2021-11-25 14:58
 */
public class BoxState implements IState<BoxState> {

    private float[] stateData;

    public BoxState(float[] stateData) {
        this.stateData = stateData;
    }

    @Override
    public BoxState clone() {
        return new BoxState(stateData.clone());
    }

    @Override
    public Class<BoxStateCollector> getCollectorClz() {
        return BoxStateCollector.class;
    }

    @Override
    public NDList singleStateList(NDManager manager) {
        return new NDList(manager.create(new float[][]{stateData}));
    }

    @Override
    public String toString() {
        return "BoxState{" +
                "stateData=" + Arrays.toString(stateData) +
                '}';
    }

    public float[] getStateData() {
        return stateData;
    }
}
