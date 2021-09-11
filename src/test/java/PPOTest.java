import agent.PPO;
import ai.djl.engine.Engine;
import env.common.Environment;
import env.demo.mountaincar.MountainCar;
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
        Environment env = new MountainCar(false);
        env.seed(0);
        new Runner(new PPO(env.getStateSpaceDim(), env.getActionSpaceDim(), 64, 0.99f, 0.95f, 0.001f, 16, 8, 0.2f), env)
                .mainLoop(100, 1000);
    }
}
