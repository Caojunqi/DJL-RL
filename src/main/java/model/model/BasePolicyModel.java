package model.model;

import env.common.action.Action;

/**
 * 策略模型基类
 *
 * @author Caojunqi
 * @date 2021-09-23 22:00
 */
public abstract class BasePolicyModel<A extends Action> extends BaseModel {

    /**
     * 根据指定环境状态选择一个合适的动作
     *
     * @param state 环境当前状态
     * @return action 接下来应采取的动作
     */
    public abstract A selectAction(float[] state);

    /**
     * 采用贪婪策略，为指定环境状态选择一个确定的动作
     *
     * @param state 环境当前状态
     * @return 确定动作
     */
    public abstract A greedyAction(float[] state);

}
