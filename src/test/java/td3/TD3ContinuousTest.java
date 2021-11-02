package td3;

import ai.djl.engine.Engine;
import algorithm.td3.TD3Continuous;
import env.demo.pendulum.Pendulum;
import utils.Runner;

/**
 * SAC算法测试类
 *
 * @author Caojunqi
 * @date 2021-10-12 15:02
 */
public class TD3ContinuousTest {

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(0);
        Pendulum env = new Pendulum();
        TD3Continuous algorithm = new TD3Continuous(env.getStateSpaceDim(), env.getActionSpaceDim());
        env.seed(0);
        new Runner<>(env, algorithm)
                .mainLoop();
    }
}
