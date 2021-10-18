package utils.datatype;

import ai.djl.ndarray.NDList;
import env.common.action.Action;

import java.util.Objects;

/**
 * 策略模型执行结果
 *
 * @author Caojunqi
 * @date 2021-10-18 12:26
 */
public class PolicyPair<A extends Action> {

    private final A action;
    private final NDList info;

    public PolicyPair(A action, NDList info) {
        this.action = action;
        this.info = info;
    }

    public static <A extends Action> PolicyPair<A> of(A action, NDList info) {
        return new PolicyPair<>(action, info);
    }

    public String toString() {
        return "PolicyPair[" + action + "," + info + "]";
    }

    public boolean equals(Object other) {
        return
                other instanceof PolicyPair<?> &&
                        Objects.equals(action, ((PolicyPair<?>) other).action) &&
                        Objects.equals(info, ((PolicyPair<?>) other).info);
    }

    public int hashCode() {
        if (action == null) return (info == null) ? 0 : info.hashCode() + 1;
        else if (info == null) return action.hashCode() + 2;
        else return action.hashCode() * 17 + info.hashCode();
    }

    public A getAction() {
        return action;
    }

    public NDList getInfo() {
        return info;
    }
}
