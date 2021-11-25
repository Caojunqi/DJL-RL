package sac;

import ai.djl.engine.Engine;
import algorithm.sac.SACContinuous;
import env.demo.pendulum.Pendulum;
import env.state.core.impl.BoxState;
import utils.Runner;

/**
 * SAC算法测试类
 *
 * @author Caojunqi
 * @date 2021-10-12 15:02
 */
public class SACContinuousTest {

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(0);
        Pendulum env = new Pendulum();
        SACContinuous<BoxState> algorithm = new SACContinuous<>(env);
        env.seed(0);
        new Runner<>(env, algorithm)
                .mainLoop();
    }
}
