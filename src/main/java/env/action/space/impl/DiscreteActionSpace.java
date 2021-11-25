package env.action.space.impl;

import env.action.core.impl.DiscreteAction;
import env.action.space.IActionSpace;

/**
 * 离散型动作空间
 *
 * @author Caojunqi
 * @date 2021-09-13 12:02
 */
public class DiscreteActionSpace implements IActionSpace<DiscreteAction> {

    /**
     * 离散型元素数目
     */
    private int num;

    public DiscreteActionSpace(int num) {
        this.num = num;
    }

    @Override
    public boolean canStep(DiscreteAction action) {
        int actionData = action.getActionData();
        return actionData >= 0 && actionData < num;
    }

    @Override
    public int getDim() {
        return num;
    }

}
