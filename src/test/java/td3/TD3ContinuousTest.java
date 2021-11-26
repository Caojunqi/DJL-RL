package td3;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import algorithm.td3.TD3Continuous;
import demo.pendulum.Pendulum;
import env.state.core.impl.BoxState;
import utils.Runner;

/**
 * SAC算法测试类
 *
 * @author Caojunqi
 * @date 2021-10-12 15:02
 */
public class TD3ContinuousTest {

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();
        Engine.getInstance().setRandomSeed(0);
        Pendulum env = new Pendulum();
        TD3Continuous<BoxState> algorithm = new TD3Continuous<>(manager, env);
        env.seed(0);
        new Runner<>(env, algorithm)
                .mainLoop();
    }
}
