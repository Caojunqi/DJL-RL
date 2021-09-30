package utils;

import agent.BaseAgent;
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
    private final BaseAgent<A, E> agent;
    private final Environment<A> env;

    public Runner(BaseAgent<A, E> agent, Environment<A> env) {
        this.agent = agent;
        this.env = env;
    }

    public void mainLoop(int maxIterNum, int minBatchSize) {
        for (int i = 0; i < maxIterNum; i++) {
            collectSamples(minBatchSize);
            agent.updateModel();
            System.out.println("完成===" + i);
        }
    }

    /**
     * 收集样本数据
     *
     * @param minBatchSize 每次刷新参数，最少所需样本数量
     */
    private void collectSamples(int minBatchSize) {
        agent.resetMemory();
        int sampleNum = 0;
        int episodesNum = 0;
        float totalReward = 0;
        float minEpisodeReward = Float.MAX_VALUE;
        float maxEpisodeReward = Float.MIN_VALUE;
        while (sampleNum < minBatchSize) {
            float[] state = env.reset().clone();
            boolean done = false;
            int step = 0;
            float episodeReward = 0;

            while (!done) {
                env.render();
                A action = agent.selectAction(state);
                Snapshot snapshot = env.step(action);
                agent.collect(state, action, snapshot.isDone(), snapshot.getNextState(), snapshot.getReward());

                done = snapshot.isDone();
                state = snapshot.getNextState().clone();
                step++;

                episodeReward += snapshot.getReward();
            }

            totalReward += episodeReward;
            minEpisodeReward = Math.min(minEpisodeReward, episodeReward);
            maxEpisodeReward = Math.min(maxEpisodeReward, episodeReward);

            sampleNum += step;
            episodesNum += 1;
        }

        System.out.println("AverageEpisodeReward [" + (totalReward / episodesNum) + "] MaxEpisodeReward [" + maxEpisodeReward + "] MinEpisodeReward [" + minEpisodeReward + "]");
    }
}
