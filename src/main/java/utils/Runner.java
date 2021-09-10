package utils;

import agent.base.BaseAgent;
import env.Environment;
import utils.datatype.Snapshot;

/**
 * RL算法执行器
 *
 * @author Caojunqi
 * @date 2021-09-09 21:59
 */
public class Runner {
    private final BaseAgent agent;
    private final Environment env;

    public Runner(BaseAgent agent, Environment env) {
        this.agent = agent;
        this.env = env;
    }

    /**
     * 执行RL算法，达到目标分数后停止
     *
     * @param goal 目标分数
     */
    public void run(double goal) {
        double score = Double.NEGATIVE_INFINITY;
        int episode = 0;

        while (score < goal) {
            episode++;
            Snapshot snapshot = env.reset();
            boolean done = false;
            int episodeScore = 0;

            while (!done) {
                env.render();
                snapshot = env.step(agent.react(snapshot.getState()));
                done = snapshot.isDone();
                episodeScore += snapshot.getReward();
                agent.collect(snapshot.getReward(), done);
            }
            score = score > Double.NEGATIVE_INFINITY ? score * 0.95 + episodeScore * 0.05 : episodeScore;
            System.out.printf("Episode %d (%d): %.2f\n", episode, episodeScore, score);
        }
    }
}
