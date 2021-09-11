package utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import utils.datatype.Transition;

/**
 * 小批量样本数据
 * 每个MemoryBatch由5部分数据组成，按顺序依次为[state, action, mask, nextState, reward]，
 * 各部分数据的长度一致，各部分数据相同index的数值抽取出来可以组成一个样本数据。
 * 各数据含义同{@link Transition}一致
 *
 * @author Caojunqi
 * @date 2021-09-11 11:16
 */
public final class MemoryBatch extends NDList {

    public MemoryBatch(NDArray... arrays) {
        super(arrays);
    }

    public NDArray getStates() {
        return get(0);
    }

    public NDArray getActions() {
        return get(1);
    }

    public NDArray getMasks() {
        return get(2);
    }

    public NDArray getNextStates() {
        return get(3);
    }

    public NDArray getRewards() {
        return get(4);
    }

}
