import env.common.action.impl.DiscreteAction;
import env.demo.cartpole.CartPole;

/**
 * 测试类
 *
 * @author Caojunqi
 * @date 2021-09-10 10:06
 */
public class MainTest {

    public static void main(String[] args) {
        CartPole cartPole = new CartPole(false);
        float[] state = new float[]{0.03073904f, 0.00145001f, -0.03088818f, -0.03131252f};
        cartPole.testReset(state);
        cartPole.step(new DiscreteAction(1));
        System.out.println(state);
    }

}
