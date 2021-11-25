package env.state.collector;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import env.state.core.IState;

/**
 * 状态数据处理器接口
 *
 * @author Caojunqi
 * @date 2021-11-25 14:55
 */
public interface IStateCollector {

    /**
     * 收集指定状态数据
     */
    void addState(int index, IState state);

    /**
     * 将数据收集结果构建成一个NDArray
     *
     * @return 转换结果
     */
    NDArray createNDArray(NDManager manager);
}
