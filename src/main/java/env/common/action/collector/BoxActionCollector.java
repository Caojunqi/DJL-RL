package env.common.action.collector;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import env.common.action.Action;
import env.common.action.IActionCollector;
import env.common.action.impl.BoxAction;

import java.util.ArrayList;
import java.util.List;

/**
 * 连续型动作数据收集器
 *
 * @author Caojunqi
 * @date 2021-09-15 11:45
 */
public class BoxActionCollector implements IActionCollector {

    private List<BoxAction> actions = new ArrayList<>();

    @Override
    public void addAction(Action action) {
        this.actions.add((BoxAction) action);
    }

    @Override
    public NDArray createNDArray(NDManager manager) {
        double[][] data = new double[actions.size()][];
        for (int i = 0; i < actions.size(); i++) {
            data[i] = actions.get(i).getActionData();
        }
        return manager.create(data);
    }
}
