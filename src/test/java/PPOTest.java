import agent.CartPoleAgent;
import ai.djl.engine.Engine;
import algorithm.AlgorithmType;
import env.demo.cartpole.CartPole;
import utils.Runner;

/**
 * PPO算法测试类
 *
 * @author Caojunqi
 * @date 2021-09-09 20:57
 */
public class PPOTest {

    public static void main(String[] args) {
//        Engine.getInstance().setRandomSeed(0);
//        MountainCarContinuous env = new MountainCarContinuous(false);
//        env.seed(0);
//        new Runner<>(new MountainCarContinuousAgent(env, AlgorithmType.PPO_CONTINUOUS), env)
//                .mainLoop();

        Engine.getInstance().setRandomSeed(0);
        CartPole env = new CartPole(false);
        env.seed(0);
        new Runner<>(new CartPoleAgent(env, AlgorithmType.PPO_DISCRETE), env)
                .mainLoop();
    }
}
