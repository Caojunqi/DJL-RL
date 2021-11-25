package env.action.space.impl;

import env.action.core.impl.MultiDiscreteAction;
import env.action.space.IActionSpace;
import org.apache.commons.lang3.Validate;

/**
 * 多维度离散型动作空间
 *
 * @author Caojunqi
 * @date 2021-09-13 12:03
 */
public class MultiDiscreteActionSpace implements IActionSpace<MultiDiscreteAction> {

    /**
     * 离散型数据的取值，该数组长度表示数据维度，数据值表示对应维度的离散数据取值最大值（不含）。
     */
    private int[] counts;

    public MultiDiscreteActionSpace(int[] counts) {
        Validate.isTrue(counts != null, "multi-discrete action space data is invalid!!");
        this.counts = counts;
    }

    @Override
    public boolean canStep(MultiDiscreteAction action) {
        int[] actionData = action.getActionData();
        for (int i = 0; i < actionData.length; i++) {
            if (actionData[i] < 0 || actionData[i] >= counts[i]) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int getDim() {
        return this.counts.length;
    }

}
