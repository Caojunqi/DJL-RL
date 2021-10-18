import ai.djl.engine.Engine;
import algorithm.ppo.PPOContinuous;
import env.demo.mountaincar.MountainCarContinuous;
import utils.Runner;

/**
 * PPO算法测试类
 *
 * @author Caojunqi
 * @date 2021-09-09 20:57
 */
public class PPOTest {

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(0);
        MountainCarContinuous env = new MountainCarContinuous(false);
        PPOContinuous algorithm = new PPOContinuous(env.getStateSpaceDim(), env.getActionSpaceDim());
        env.seed(0);
        new Runner<>(env, algorithm)
                .mainLoop();

//        Engine.getInstance().setRandomSeed(0);
//        CartPole env = new CartPole(false);
//        PPODiscrete algorithm = new PPODiscrete(env.getStateSpaceDim(), env.getActionSpaceDim());
//        env.seed(0);
//        new Runner<>(env, algorithm)
//                .mainLoop();
    }
}
