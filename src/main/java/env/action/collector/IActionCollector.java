package env.action.collector;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import env.action.core.IAction;

/**
 * 动作数据处理器接口
 *
 * @author Caojunqi
 * @date 2021-09-15 11:41
 */
public interface IActionCollector {

    /**
     * 收集指定动作数据
     */
    void addAction(int index, IAction action);

    /**
     * 将数据收集结果构建成一个NDArray
     *
     * @return 转换结果
     */
    NDArray createNDArray(NDManager manager);
}
