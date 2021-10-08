package agent;

import ai.djl.ndarray.NDManager;
import algorithm.AlgorithmType;
import algorithm.BaseAlgorithm;
import env.common.Environment;
import env.common.action.Action;
import model.model.BasePolicyModel;
import model.model.BaseValueModel;
import utils.Memory;

import java.util.Random;

/**
 * RL算法调度器基类
 *
 * @author Caojunqi
 * @date 2021-09-09 21:57
 */
public abstract class BaseAgent<A extends Action, E extends Environment<A>> {
    /**
     * 随机数生成器
     */
    protected Random random = new Random(0);
    /**
     * 调度器针对的环境
     */
    private E env;
    /**
     * 样本缓存
     */
    protected Memory<A> memory = new Memory<>();
    /**
     * NDArray管理器
     */
    protected NDManager manager;
    /**
     * 算法核心
     */
    protected BaseAlgorithm<A> algorithm;
    /**
     * 策略模型
     */
    protected BasePolicyModel<A> policyModel;
    /**
     * 价值函数近似模型
     */
    protected BaseValueModel valueModel;

    protected BaseAgent(E env, AlgorithmType algorithmType) {
        this.env = env;
        manager = NDManager.newBaseManager();
        this.algorithm = algorithmType.getAlgorithm();
    }

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
    public final void updateModel() {
        this.algorithm.updateModel(manager, memory, policyModel, valueModel);
    }

    /**
     * 重置样本缓存
     */
    public void resetMemory() {
        this.memory.reset();
    }

    public BasePolicyModel<A> getPolicyModel() {
        return policyModel;
    }

    public BaseValueModel getValueModel() {
        return valueModel;
    }
}
