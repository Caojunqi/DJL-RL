package sac;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import algorithm.sac.SACDiscrete;
import demo.cartpole.CartPole;
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
        NDManager manager = NDManager.newBaseManager();
        Engine.getInstance().setRandomSeed(0);
        CartPole env = new CartPole(false);
        SACDiscrete<BoxState> algorithm = new SACDiscrete<>(manager, env);
        env.seed(0);
        new Runner<>(env, algorithm)
                .mainLoop();

    }
}
