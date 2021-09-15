package env.common.action.impl;

import env.common.action.Action;
import env.common.action.collector.MultiDiscreteActionCollector;
import env.common.spaces.action.MultiDiscreteActionSpace;

import java.util.Arrays;

/**
 * 针对{@link MultiDiscreteActionSpace}型动作空间的Action数据
 *
 * @author Caojunqi
 * @date 2021-09-13 11:26
 */
public class MultiDiscreteAction implements Action {

    private int[] actionData;

    public MultiDiscreteAction(int[] actionData) {
        this.actionData = actionData;
    }

    @Override
    public Class<?> getCollectorClz() {
        return MultiDiscreteActionCollector.class;
    }

    @Override
    public String toString() {
        return "MultiDiscreteAction{" +
                "actionData=" + Arrays.toString(actionData) +
                '}';
    }

    public int[] getActionData() {
        return actionData;
    }
}
