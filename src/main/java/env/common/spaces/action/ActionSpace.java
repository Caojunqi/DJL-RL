package env.common.spaces.action;

import env.common.action.Action;

/**
 * 动作空间接口
 *
 * @author Caojunqi
 * @date 2021-09-13 11:59
 */
public interface ActionSpace<A extends Action> {

    /**
     * 检验当前动作空间是否可以执行指定行为
     *
     * @param action 待执行的行为
     * @return true-可以执行；false-不可执行
     */
    boolean canStep(A action);

    /**
     * 动作空间维度
     */
    int getDim();
}
