import agent.CartPoleAgent;
import ai.djl.engine.Engine;
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
//        new Runner<>(new MountainCarContinuousAgent(env, 0.99f, 0.95f, 3e-4f, 10, 64, 0.2f), env)
//                .mainLoop(500, 2048);

        Engine.getInstance().setRandomSeed(0);
        CartPole env = new CartPole(false);
        env.seed(0);
        new Runner<>(new CartPoleAgent(env, 0.99f, 0.95f, 3e-4f, 10, 64, 0.2f), env)
                .mainLoop(500, 2048);
    }
}
