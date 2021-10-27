package utils.datatype;

import ai.djl.ndarray.NDList;
import algorithm.BaseAlgorithm;
import env.common.action.Action;

import java.util.List;
import java.util.Objects;

/**
 * 策略模型执行结果
 *
 * @author Caojunqi
 * @date 2021-10-18 12:26
 */
public class PolicyPair<A extends Action> {

    private final List<A> actions;
    private final NDList info;

    private PolicyPair(List<A> actions, NDList info) {
        this.actions = actions;
        this.info = info;
    }

    public static <A extends Action> PolicyPair<A> of(List<A> action, NDList info) {
        return new PolicyPair<>(action, info);
    }

    public String toString() {
        return "PolicyPair[" + actions + "," + info + "]";
    }

    public boolean equals(Object other) {
        return
                other instanceof PolicyPair<?> &&
                        Objects.equals(actions, ((PolicyPair<?>) other).actions) &&
                        Objects.equals(info, ((PolicyPair<?>) other).info);
    }

    public int hashCode() {
        if (actions == null) return (info == null) ? 0 : info.hashCode() + 1;
        else if (info == null) return actions.hashCode() + 2;
        else return actions.hashCode() * 17 + info.hashCode();
    }

    /**
     * 判定actions中只有一个元素，并返回该元素；
     * 该接口只能用于{@link BaseAlgorithm#selectAction(float[])}和{@link BaseAlgorithm#greedyAction(float[])}接口中，
     * 也即只用于针对一个状态来获取对应的行为。
     */
    public A singletonOrThrow() {
        if (actions.size() != 1) {
            throw new IndexOutOfBoundsException(
                    "Incorrect number of elements in PolicyPair.singletonOrThrow: Expected 1 and was "
                            + actions.size());
        }
        return actions.get(0);
    }

    public List<A> getActions() {
        return actions;
    }

    public NDList getInfo() {
        return info;
    }
}
