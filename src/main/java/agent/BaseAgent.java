package agent;

import ai.djl.translate.TranslateException;
import env.common.Environment;
import env.common.action.Action;
import utils.Memory;

/**
 * RL算法调度器基类
 *
 * @author Caojunqi
 * @date 2021-09-09 21:57
 */
public abstract class BaseAgent<A extends Action, E extends Environment<A>> {
    /**
     * 调度器针对的环境
     */
    private E env;
    /**
     * 样本缓存
     */
    protected Memory<A> memory = new Memory<>();

    protected BaseAgent(E env) {
        this.env = env;
    }

    /**
     * 根据指定环境状态选择一个合适的动作
     *
     * @param state 环境当前状态
     * @return action 接下来应采取的动作
     */
    public abstract A selectAction(float[] state);

    /**
     * 收集动作执行样本数据
     *
     * @param state     动作执行之前的环境状态信息
     * @param action    所执行的动作
     * @param done      当前幕是否已结束，true-结束，false-未结束
     * @param nextState 动作执行之后的环境状态信息
     * @param reward    之前的动作获取到的收益
     */
    public abstract void collect(float[] state, A action, boolean done, float[] nextState, float reward);

    /**
     * 更新模型参数
     */
    public abstract void updateModel() throws TranslateException;

    /**
     * 重置样本缓存
     */
    public void resetMemory() {
        this.memory.reset();
    }
}
