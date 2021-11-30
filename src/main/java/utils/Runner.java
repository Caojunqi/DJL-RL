package utils;

import algorithm.BaseAlgorithm;
import algorithm.CommonParameter;
import env.action.core.IAction;
import env.common.Environment;
import env.state.core.IState;
import utils.datatype.Snapshot;

/**
 * RL算法执行器
 *
 * @author Caojunqi
 * @date 2021-09-09 21:59
 */
public class Runner<S extends IState<S>, A extends IAction> {
    private final Environment<S, A> env;
    private final BaseAlgorithm<S, A> algorithm;

    public Runner(Environment<S, A> env, BaseAlgorithm<S, A> algorithm) {
        this.env = env;
        this.algorithm = algorithm;
    }

    public void mainLoop() {
        for (int i = 0; i < CommonParameter.MAX_ITER_NUM; i++) {
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
        while (sampleNum < CommonParameter.MIN_BATCH_SIZE) {
            S state = env.reset().clone();
            boolean done = false;
            int step = 0;
            float episodeReward = 0;

            while (!done) {
                env.render();
                A action = algorithm.selectAction(state);
                Snapshot<S> snapshot = env.step(action);
                algorithm.collect(state, action, snapshot.isDone(), snapshot.getNextState().clone(), snapshot.getReward());

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

        S state = env.reset();
        boolean done = false;
        float episodeReward = 0;

        while (!done) {
            env.render();
            A action = algorithm.greedyAction(state);
            Snapshot<S> snapshot = env.step(action);
            algorithm.collect(state, action, snapshot.isDone(), snapshot.getNextState(), snapshot.getReward());

            episodeReward += snapshot.getReward();

            // System.out.println("TestModel=====State[" + Arrays.toString(state) + "]  Action[" + action.toString() + "]");

            done = snapshot.isDone();
            state = snapshot.getNextState();
        }

        System.out.println("TestModel=====EpisodeReward [" + episodeReward + "]");
    }
}
