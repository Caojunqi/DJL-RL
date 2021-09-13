import env.common.spaces.MultiDiscrete;
import env.common.spaces.Space;

/**
 * 测试类
 *
 * @author Caojunqi
 * @date 2021-09-10 10:06
 */
public class MainTest {

    public static void main(String[] args) {
        int[] STATE_SPACE = new int[]{2, 10};
        Space space = new MultiDiscrete(STATE_SPACE);
        System.out.println(space.contains(new double[]{1.2, 3.4}));
    }
}
