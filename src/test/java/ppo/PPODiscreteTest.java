package ppo;

import ai.djl.engine.Engine;
import algorithm.ppo.PPODiscrete;
import env.demo.cartpole.CartPole;
import env.state.core.impl.BoxState;
import utils.Runner;

/**
 * PPO算法测试类
 *
 * @author Caojunqi
 * @date 2021-09-09 20:57
 */
public class PPODiscreteTest {

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(0);
        CartPole env = new CartPole(false);
        PPODiscrete<BoxState> algorithm = new PPODiscrete<>(env);
        env.seed(0);
        new Runner<>(env, algorithm)
                .mainLoop();
    }
}
