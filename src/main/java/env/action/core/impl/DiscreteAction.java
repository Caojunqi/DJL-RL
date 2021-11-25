package env.action.core.impl;

import env.action.collector.impl.DiscreteActionCollector;
import env.action.core.IAction;
import env.action.space.impl.DiscreteActionSpace;

/**
 * 针对{@link DiscreteActionSpace}型动作空间的Action数据
 *
 * @author Caojunqi
 * @date 2021-09-13 11:25
 */
public class DiscreteAction implements IAction {

    private int actionData;

    public DiscreteAction(int actionData) {
        this.actionData = actionData;
    }

    @Override
    public Class<DiscreteActionCollector> getCollectorClz() {
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
