package env.state.core;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import env.state.collector.IStateCollector;

/**
 * 状态数据接口
 *
 * @author Caojunqi
 * @date 2021-11-25 14:53
 */
public interface IState<S extends IState<S>> extends Cloneable {

    S clone();

    /**
     * 返回对应的状态数据收集器类型
     */
    Class<? extends IStateCollector> getCollectorClz();

    /**
     * 将单个状态数据转为一个NDList，用作神经网络输入
     */
    NDList singleStateList(NDManager manager);
}
