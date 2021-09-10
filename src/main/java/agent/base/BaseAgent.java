package agent.base;

/**
 * RL算法调度器基类
 *
 * @author Caojunqi
 * @date 2021-09-09 21:57
 */
public abstract class BaseAgent {

    /**
     * 根据指定环境状态计算出一个合适的动作
     *
     * @param state 环境当前状态
     * @return action 接下来应采取的动作
     */
    public abstract int react(float[] state);

    /**
     * 把之前动作执行的结果收集起来
     *
     * @param reward 之前的动作获取到的收益
     * @param done   当前幕是否已结束，true-结束，false-未结束
     */
    public abstract void collect(float reward, boolean done);
}
