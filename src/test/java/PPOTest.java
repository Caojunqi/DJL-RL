import ai.djl.engine.Engine;
import env.common.Environment;
import env.demo.mountaincar.MountainCar;

/**
 * PPO算法测试类
 *
 * @author Caojunqi
 * @date 2021-09-09 20:57
 */
public class PPOTest {

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(0);
        Environment env = new MountainCar();
    }
}
