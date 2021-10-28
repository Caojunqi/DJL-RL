package sac;

import ai.djl.engine.Engine;
import algorithm.sac.SACContinuous;
import env.demo.mountaincar.MountainCarContinuous;
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
        MountainCarContinuous env = new MountainCarContinuous(false);
        SACContinuous algorithm = new SACContinuous(env.getStateSpaceDim(), env.getActionSpaceDim());
        env.seed(0);
        new Runner<>(env, algorithm)
                .mainLoop();
    }
}
