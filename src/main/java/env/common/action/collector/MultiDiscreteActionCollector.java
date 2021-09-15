package env.common.action.collector;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import env.common.action.Action;
import env.common.action.IActionCollector;
import env.common.action.impl.MultiDiscreteAction;

import java.util.ArrayList;
import java.util.List;

/**
 * 多维离散型动作数据收集器
 *
 * @author Caojunqi
 * @date 2021-09-15 15:54
 */
public class MultiDiscreteActionCollector implements IActionCollector {

    private List<MultiDiscreteAction> actions = new ArrayList<>();

    @Override
    public void addAction(Action action) {
        this.actions.add((MultiDiscreteAction) action);
    }

    @Override
    public NDArray createNDArray(NDManager manager) {
        int[][] data = new int[actions.size()][];
        for (int i = 0; i < actions.size(); i++) {
            data[i] = actions.get(i).getActionData();
        }
        return manager.create(data);
    }
}
