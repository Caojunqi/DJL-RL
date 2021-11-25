package env.demo.cartpole;

import env.action.core.impl.DiscreteAction;
import env.action.space.impl.DiscreteActionSpace;
import env.common.Environment;
import env.state.core.impl.BoxState;
import env.state.space.impl.BoxStateSpace;
import org.apache.commons.lang3.Validate;
import utils.datatype.Snapshot;

/*_
 * 复刻gym中的CartPole-v1环境
 *
 *     Description:
 *         A pole is attached by an un-actuated joint to a cart, which moves along
 *         a frictionless track. The pendulum starts upright, and the goal is to
 *         prevent it from falling over by increasing and reducing the cart's
 *         velocity.
 *
 *     Source:
 *         This environment corresponds to the version of the cart-pole problem
 *         described by Barto, Sutton, and Anderson
 *
 *     Observation:
 *         Type: Box(4)
 *         Num     Observation               Min                     Max
 *         0       Cart Position             -4.8                    4.8
 *         1       Cart Velocity             -Inf                    Inf
 *         2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
 *         3       Pole Angular Velocity     -Inf                    Inf
 *
 *     Actions:
 *         Type: Discrete(2)
 *         Num   Action
 *         0     Push cart to the left
 *         1     Push cart to the right
 *
 *         Note: The amount the velocity that is reduced or increased is not
 *         fixed; it depends on the angle the pole is pointing. This is because
 *         the center of gravity of the pole increases the amount of energy needed
 *         to move the cart underneath it
 *
 *     Reward:
 *         Reward is 1 for every step taken, including the termination step
 *
 *     Starting State:
 *         All observations are assigned a uniform random value in [-0.05..0.05]
 *
 *     Episode Termination:
 *         Pole Angle is more than 12 degrees.
 *         Cart Position is more than 2.4 (center of the cart reaches the edge of
 *         the display).
 *         Episode length is greater than 500.
 *         Solved Requirements:
 *         Considered solved when the average return is greater than or equal to
 *         195.0 over 100 consecutive trials.
 *
 * @author Caojunqi
 * @date 2021-09-09 21:07
 */
public class CartPole extends Environment<BoxState, DiscreteAction> {
    private static final double[][] STATE_SPACE = new double[][]{{-4.8, 4.8},
            {Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY}, {-0.418, 0.418},
            {Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY}};
    private static final double GRAVITY = 9.8;
    private static final double CART_MASS = 1.0;
    private static final double POLE_MASS = 0.1;
    private static final double TOTAL_MASS = CART_MASS + POLE_MASS;
    private static final double LENGTH = 0.5;
    private static final double POLEMASS_LENGTH = POLE_MASS * LENGTH;
    private static final double FORCE_MAG = 10.0;
    private static final double TAU = 0.02;
    private static final double X_THRESHOLD = 2.4;
    private static final double THETA_THRESHOLD = 12 * 2 * Math.PI / 360;
    private static final double MAX_EPISODE_LENGTH = 500;

    private final float[] state = new float[]{0.0f, 0.0f, 0.0f, 0.0f};
    private final CartPoleVisualizer visualizer;

    public int count = 0;

    public CartPole(boolean visual) {
        super(new BoxStateSpace(STATE_SPACE), new DiscreteActionSpace(2));
        visualizer = visual ? new CartPoleVisualizer(LENGTH, X_THRESHOLD, 1000) : null;
    }

    @Override
    protected Snapshot<BoxState> doStep(DiscreteAction action) {
        Validate.isTrue(actionSpace.canStep(action), "action[" + action + "] invalid!!");
        int actionData = action.getActionData();
        double force = actionData == 1 ? FORCE_MAG : -FORCE_MAG;
        double cos_theta = Math.cos(state[2]);
        double sin_theta = Math.sin(state[2]);
        double temp = (force + POLEMASS_LENGTH * Math.pow(state[3], 2) * sin_theta) / TOTAL_MASS;

        double theta_acc = ((GRAVITY * sin_theta - temp * cos_theta)
                / (LENGTH * (4.0 / 3.0 - POLE_MASS * Math.pow(cos_theta, 2) / TOTAL_MASS)));
        double x_acc = temp - POLEMASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

        state[0] += TAU * state[1];
        state[1] += TAU * x_acc;
        state[2] += TAU * state[3];
        if (state[2] > Math.PI) {
            state[2] -= 2 * Math.PI;
        } else if (state[2] < -Math.PI) {
            state[2] += 2 * Math.PI;
        }
        state[3] += TAU * theta_acc;
        boolean done = (state[0] < -X_THRESHOLD || state[0] > X_THRESHOLD || state[2] < -THETA_THRESHOLD
                || state[2] > THETA_THRESHOLD);

        return new Snapshot<>(new BoxState(state), 1.0f, count++ > MAX_EPISODE_LENGTH || done);
    }

    @Override
    public BoxState reset() {
        for (int i = 0; i < 4; i++) {
            state[i] = random.nextFloat() * 0.1f - 0.05f;
        }
        count = 0;
        return new BoxState(state);
    }

    @Override
    public void render() {
        if (visualizer != null) {
            visualizer.update(state);
        }
    }

    @Override
    public void close() {

    }
}
