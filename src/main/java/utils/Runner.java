package utils;

import algorithm.BaseAlgorithm;
import env.common.Environment;
import env.common.action.Action;
import resource.ConstantParameter;
import utils.datatype.Snapshot;

/**
 * RL算法执行器
 *
 * @author Caojunqi
 * @date 2021-09-09 21:59
 */
public class Runner<A extends Action, E extends Environment<A>> {
    private final Environment<A> env;
    private final BaseAlgorithm<A> algorithm;

    public Runner(Environment<A> env, BaseAlgorithm<A> algorithm) {
        this.env = env;
        this.algorithm = algorithm;
    }

    public void mainLoop() {
        for (int i = 0; i < ConstantParameter.MAX_ITER_NUM; i++) {
            collectSamples();
            algorithm.updateModel();
            testModel();
            System.out.println("完成===" + i);
        }
    }

    /**
     * 收集样本数据
     */
    private void collectSamples() {
        algorithm.resetMemory();
        int sampleNum = 0;
        int episodesNum = 0;
        float totalReward = 0;
        float minEpisodeReward = Float.POSITIVE_INFINITY;
        float maxEpisodeReward = Float.NEGATIVE_INFINITY;
        while (sampleNum < ConstantParameter.MIN_BATCH_SIZE) {
            float[] state = env.reset().clone();
            boolean done = false;
            int step = 0;
            float episodeReward = 0;

            while (!done) {
                env.render();
                A action = algorithm.selectAction(state);
                Snapshot snapshot = env.step(action);
                algorithm.collect(state, action, snapshot.isDone(), snapshot.getNextState(), snapshot.getReward());

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
        algorithm.resetMemory();

        float[] state = env.reset().clone();
        boolean done = false;
        float episodeReward = 0;

        while (!done) {
            env.render();
            A action = algorithm.greedyAction(state);
            Snapshot snapshot = env.step(action);
            algorithm.collect(state, action, snapshot.isDone(), snapshot.getNextState(), snapshot.getReward());

            episodeReward += snapshot.getReward();

            // System.out.println("TestModel=====State[" + Arrays.toString(state) + "]  Action[" + action.toString() + "]");

            done = snapshot.isDone();
            state = snapshot.getNextState().clone();
        }

        System.out.println("TestModel=====EpisodeReward [" + episodeReward + "]");
    }
}
