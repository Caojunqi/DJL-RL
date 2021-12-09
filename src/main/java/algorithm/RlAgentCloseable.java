package algorithm;

import ai.djl.modality.rl.agent.RlAgent;

/**
 * 可执行关闭操作的强化学习智能体
 *
 * @author Caojunqi
 * @date 2021-12-09 11:52
 */
public interface RlAgentCloseable extends RlAgent {

    /**
     * 智能体关闭，清除智能体所消耗的资源
     */
    void close();
}
