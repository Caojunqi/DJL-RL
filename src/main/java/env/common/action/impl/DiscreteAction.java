package env.common.action.impl;

import env.common.action.Action;
import env.common.action.collector.DiscreteActionCollector;
import env.common.spaces.action.DiscreteActionSpace;

/**
 * 针对{@link DiscreteActionSpace}型动作空间的Action数据
 *
 * @author Caojunqi
 * @date 2021-09-13 11:25
 */
public class DiscreteAction implements Action {

    private int actionData;

    public DiscreteAction(int actionData) {
        this.actionData = actionData;
    }

    @Override
    public Class<?> getCollectorClz() {
        return DiscreteActionCollector.class;
    }

    @Override
    public String toString() {
        return "DiscreteAction{" +
                "actionData=" + actionData +
                '}';
    }

    public int getActionData() {
        return actionData;
    }
}
