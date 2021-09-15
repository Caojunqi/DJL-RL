package utils;

import agent.BaseAgent;
import ai.djl.translate.TranslateException;
import env.common.Environment;
import env.common.action.Action;
import utils.datatype.Snapshot;

/**
 * RL算法执行器
 *
 * @author Caojunqi
 * @date 2021-09-09 21:59
 */
public class Runner<A extends Action, E extends Environment<A>> {
    /**
     * 一幕允许的最大步数
     */
    private static final int MAX_STEP_ONE_EPISODE = 10000;

    private final BaseAgent<A, E> agent;
    private final Environment<A> env;

    public Runner(BaseAgent<A, E> agent, Environment<A> env) {
        this.agent = agent;
        this.env = env;
    }

    public void mainLoop(int maxIterNum, int minBatchSize) {
        for (int i = 0; i < maxIterNum; i++) {
            collectSamples(minBatchSize);
            try {
                agent.updateModel();
            } catch (TranslateException e) {
                throw new IllegalStateException(e);
            }
            System.out.println("完成===" + i);
        }
    }

    /**
     * 收集样本数据
     *
     * @param minBatchSize 每次刷新参数，最少所需样本数量
     */
    private void collectSamples(int minBatchSize) {
        int sampleNum = 0;
        int episodesNum = 0;
        while (sampleNum < minBatchSize) {
            float[] state = env.reset();
            boolean done = false;
            int step = 0;

            while (step < MAX_STEP_ONE_EPISODE && !done) {
                env.render();
                A action = agent.selectAction(state);
                Snapshot snapshot = env.step(action);

                agent.collect(state, action, snapshot.isDone(), snapshot.getNextState(), snapshot.getReward());

                done = snapshot.isDone();
                state = snapshot.getNextState();
                step++;
            }

            sampleNum += step;
            episodesNum += 1;
        }
    }
}
