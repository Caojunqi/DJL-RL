import env.common.spaces.Box;
import env.common.spaces.Space;

/**
 * 测试类
 *
 * @author Caojunqi
 * @date 2021-09-10 10:06
 */
public class MainTest {

    public static void main(String[] args) {
        double[][] STATE_SPACE = new double[][]{{-1.2, 10.6}, {-10.07, 10.07}};
        Space space = new Box(STATE_SPACE);
        System.out.println(space.contains(new int[]{11, 2}));
    }
}
