package env.common.action.impl;

import env.common.action.Action;
import env.common.action.collector.BoxActionCollector;
import env.common.spaces.action.BoxActionSpace;

import java.util.Arrays;

/**
 * 针对{@link BoxActionSpace}型动作空间的Action数据
 *
 * @author Caojunqi
 * @date 2021-09-13 11:24
 */
public class BoxAction implements Action {

    private float[] actionData;

    public BoxAction(float[] actionData) {
        this.actionData = actionData;
    }

    @Override
    public Class<BoxActionCollector> getCollectorClz() {
        return BoxActionCollector.class;
    }

    @Override
    public String toString() {
        return "BoxAction{" +
                "actionData=" + Arrays.toString(actionData) +
                '}';
    }

    public float[] getActionData() {
        return actionData;
    }

}
