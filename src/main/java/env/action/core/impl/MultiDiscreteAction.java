package env.action.core.impl;

import env.action.collector.impl.MultiDiscreteActionCollector;
import env.action.core.IAction;
import env.action.space.impl.MultiDiscreteActionSpace;

import java.util.Arrays;

/**
 * 针对{@link MultiDiscreteActionSpace}型动作空间的Action数据
 *
 * @author Caojunqi
 * @date 2021-09-13 11:26
 */
public class MultiDiscreteAction implements IAction {

    private int[] actionData;

    public MultiDiscreteAction(int[] actionData) {
        this.actionData = actionData;
    }

    @Override
    public Class<MultiDiscreteActionCollector> getCollectorClz() {
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
