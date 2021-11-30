package env.state.space.impl;

import env.state.core.impl.DiscreteState;
import env.state.space.IStateSpace;

/**
 * 离散型状态空间
 *
 * @author Caojunqi
 * @date 2021-09-13 12:20
 */
public class DiscreteStateSpace implements IStateSpace<DiscreteState> {

    /**
     * 离散型元素数目
     */
    private int num;

    public DiscreteStateSpace(int num) {
        this.num = num;
    }

    @Override
    public int getDim() {
        return num;
    }

    @Override
    public int getFlatDim() {
        return num;
    }
}
