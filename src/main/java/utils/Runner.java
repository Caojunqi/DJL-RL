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
            testModel();
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
        float minEpisodeReward = Float.POSITIVE_INFINITY;
        float maxEpisodeReward = Float.NEGATIVE_INFINITY;
        while (sampleNum < minBatchSize) {
            float[] state = env.reset().clone();
            boolean done = false;
            int step = 0;
            float episodeReward = 0;

            while (!done) {
                env.render();
                A action = agent.getPolicyModel().selectAction(state);
                Snapshot snapshot = env.step(action);
                agent.collect(state, action, snapshot.isDone(), snapshot.getNextState(), snapshot.getReward());

                done = snapshot.isDone();
                state = snapshot.getNextState().clone();
                step++;

                episodeReward += snapshot.getReward();
            }

            totalReward += episodeReward;
            minEpisodeReward = Math.min(minEpisodeReward, episodeReward);
            maxEpisodeReward = Math.max(maxEpisodeReward, episodeReward);

            sampleNum += step;
            episodesNum += 1;

            System.out.println("episode[" + episodesNum + "], reward[" + episodeReward + "]");
        }

        System.out.println("AverageEpisodeReward [" + (totalReward / episodesNum) + "] MaxEpisodeReward [" + maxEpisodeReward + "] MinEpisodeReward [" + minEpisodeReward + "]");
    }

    /**
     * 采用贪婪策略，收集一幕样本数据，检测模型可靠性
     */
    private void testModel() {
        agent.resetMemory();

        float[] state = env.reset().clone();
        boolean done = false;
        float episodeReward = 0;

        while (!done) {
            env.render();
            A action = agent.getPolicyModel().greedyAction(state);
            Snapshot snapshot = env.step(action);
            agent.collect(state, action, snapshot.isDone(), snapshot.getNextState(), snapshot.getReward());

            episodeReward += snapshot.getReward();

            // System.out.println("TestModel=====State[" + Arrays.toString(state) + "]  Action[" + action.toString() + "]");

            done = snapshot.isDone();
            state = snapshot.getNextState().clone();
        }

        System.out.println("TestModel=====EpisodeReward [" + episodeReward + "]");
    }
}
