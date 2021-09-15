package env.demo.mountaincar;

import env.common.Environment;
import env.common.action.impl.DiscreteAction;
import env.common.spaces.action.DiscreteActionSpace;
import env.common.spaces.state.BoxStateSpace;
import org.apache.commons.lang3.Validate;
import utils.Helper;
import utils.datatype.Snapshot;

/*_
 * 复刻gym中的MountainCar-v0环境，动作空间是离散的
 *
 * Description:
 *     The agent (a car) is started at the bottom of a valley. For any given
 *     state the agent may choose to accelerate to the left, right or cease
 *     any acceleration.
 *
 * Source:
 *     The environment appeared first in Andrew Moore's PhD Thesis (1990).
 *
 * Observation:
 *     Type: Box(2)
 *     Num    Observation               Min            Max
 *     0      Car Position              -1.2           0.6
 *     1      Car Velocity              -0.07          0.07
 *
 * Actions:
 *     Type: Discrete(3)
 *     Num    Action
 *     0      Accelerate to the Left
 *     1      Don't accelerate
 *     2      Accelerate to the Right
 *
 *     Note: This does not affect the amount of velocity affected by the
 *     gravitational pull acting on the car.
 *
 * Reward:
 *      Reward of 0 is awarded if the agent reached the flag (position = 0.5)
 *      on top of the mountain.
 *      Reward of -1 is awarded if the position of the agent is less than 0.5.
 *
 * Starting State:
 *      The position of the car is assigned a uniform random value in
 *      [-0.6 , -0.4].
 *      The starting velocity of the car is always assigned to 0.
 *
 * Episode Termination:
 *      The car position is more than 0.5
 *      Episode length is greater than 200
 *
 * @author Caojunqi
 * @date 2021-09-09 21:28
 */
public class MountainCar extends Environment<DiscreteAction> {
    private static final double[][] STATE_SPACE = new double[][]{{-1.2, 0.6}, {-0.07, 0.07}};
    private static final float MIN_POSITION = -1.2f;
    private static final float MAX_POSITION = 0.6f;
    private static final float MIN_INITIAL_POSITION = -0.6f;
    private static final float MAX_INITIAL_POSITION = -0.4f;
    private static final float MAX_SPEED = 0.1f;
    private static final float GOAL_POSITION = 0.5f;
    private static final float GOAL_VELOCITY = 0.0f;
    private static final double FORCE = 0.001;
    private static final double GRAVITY = 0.0025;
    private static final double MAX_EPISODE_LENGTH = 200;

    private final float[] state = new float[]{0.0f, 0.0f};
    private final MountainCarVisualizer visualizer;

    private int episodeLength = 0;

    public MountainCar(boolean visual) {
        super(new BoxStateSpace(STATE_SPACE), new DiscreteActionSpace(3));
        visualizer = visual ? new MountainCarVisualizer(MIN_POSITION, MAX_POSITION, GOAL_POSITION, 1000) : null;
    }

    @Override
    protected Snapshot doStep(DiscreteAction action) {
        Validate.isTrue(actionSpace.canStep(action), "action[" + action + "] invalid!!");
        int actionData = action.getActionData();
        state[1] += (actionData - 1) * FORCE - Math.cos(3 * state[0]) * (-GRAVITY);
        state[1] = Math.min(Math.max(state[1], -MAX_SPEED), MAX_SPEED);
        state[0] += state[1];
        state[0] = Math.min(Math.max(state[0], MIN_POSITION), MAX_POSITION);

        if (state[0] == MIN_POSITION && state[1] < 0) {
            state[1] = 0;
        }
        boolean done = ((state[0] >= GOAL_POSITION && state[1] >= GOAL_VELOCITY)
                || ++episodeLength >= MAX_EPISODE_LENGTH);

        return new Snapshot(state, -1, done);
    }

    @Override
    public float[] reset() {
        episodeLength = 0;
        float initialPosition = (float) Helper.betweenDouble(MIN_INITIAL_POSITION, MAX_INITIAL_POSITION);
        this.state[0] = initialPosition;
        this.state[1] = 0;
        return this.state;
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
