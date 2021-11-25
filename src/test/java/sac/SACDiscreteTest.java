package sac;

import ai.djl.engine.Engine;
import algorithm.sac.SACDiscrete;
import env.demo.cartpole.CartPole;
import env.state.core.impl.BoxState;
import utils.Runner;

/**
 * SAC算法测试类
 *
 * @author Caojunqi
 * @date 2021-10-12 15:02
 */
public class SACDiscreteTest {

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(0);
        CartPole env = new CartPole(false);
        SACDiscrete<BoxState> algorithm = new SACDiscrete<>(env);
        env.seed(0);
        new Runner<>(env, algorithm)
                .mainLoop();

    }
}
