package env.demo.mountaincar;

import env.common.Environment;
import env.common.action.impl.BoxAction;
import env.common.spaces.action.BoxActionSpace;
import env.common.spaces.state.BoxStateSpace;
import org.apache.commons.lang3.Validate;
import utils.datatype.Snapshot;

/*_
 * 复刻gym中的MountainCarContinuous-v0环境，动作空间是连续的
 *
 * Description:
 *     The agent (a car) is started at the bottom of a valley. For any given
 *     state the agent may choose to accelerate to the left, right or cease
 *     any acceleration.
 * Observation:
 *     Type: Box(2)
 *     Num    Observation               Min            Max
 *     0      Car Position              -1.2           0.6
 *     1      Car Velocity              -0.07          0.07
 * Actions:
 *     Type: Box(1)
 *     Num    Action                    Min            Max
 *     0      the power coef            -1.0           1.0
 *     Note: actual driving force is calculated by multipling the power coef by power (0.0015)
 *
 * Reward:
 *     Reward of 100 is awarded if the agent reached the flag (position = 0.45) on top of the mountain.
 *     Reward is decrease based on amount of energy consumed each step.
 *
 * Starting State:
 *     The position of the car is assigned a uniform random value in
 *     [-0.6 , -0.4].
 *     The starting velocity of the car is always assigned to 0.
 *
 * Episode Termination:
 *     The car position is more than 0.45
 *     Episode length is greater than 200
 *
 * @author Caojunqi
 * @date 2021-09-09 21:29
 */
public class MountainCarContinuous extends Environment<BoxAction> {
    private static final double[][] STATE_SPACE = new double[][]{{-1.2, 0.6}, {-0.07, 0.07}};
    private static final double[][] ACTION_SPACE = new double[][]{{-1.0, 1.0}};
    private static final float MIN_ACTION = -1.0f;
    private static final float MAX_ACTION = 1.0f;
    private static final float MIN_POSITION = -1.2f;
    private static final float MAX_POSITION = 0.6f;
    private static final float MIN_INITIAL_POSITION = -0.6f;
    private static final float MAX_INITIAL_POSITION = -0.4f;
    private static final float MAX_SPEED = 0.1f;
    private static final float GOAL_POSITION = 0.45f;
    private static final float GOAL_VELOCITY = 0.0f;
    private static final float POWER = 0.45f;

    private final float[] state = new float[]{0.0f, 0.0f};
    private final MountainCarVisualizer visualizer;

    private int episodeLength = 0;

    public MountainCarContinuous(boolean visual) {
        super(new BoxStateSpace(STATE_SPACE), new BoxActionSpace(ACTION_SPACE));
        visualizer = visual ? new MountainCarVisualizer(MIN_POSITION, MAX_POSITION, GOAL_POSITION, 1000) : null;
    }

    @Override
    public Snapshot doStep(BoxAction action) {
        Validate.isTrue(actionSpace.canStep(action), "action[" + action + "] invalid!!");
        double[] actionData = action.getActionData();
        float position = this.state[0];
        float velocity = this.state[1];
        double force = Math.min(Math.max(actionData[0], MIN_ACTION), MAX_ACTION);
        velocity += force * POWER - 0.0025 * Math.cos(3 * position);
        if (velocity > MAX_SPEED) {
            velocity = MAX_SPEED;
        }
        if (velocity < -MAX_SPEED) {
            velocity = -MAX_SPEED;
        }
        position += velocity;
        if (position > MAX_POSITION) {
            position = MAX_POSITION;
        }
        if (position < MIN_POSITION) {
            position = MIN_POSITION;
        }
        if (position == MIN_POSITION && velocity < 0) {
            velocity = 0;
        }
        boolean done = position >= GOAL_POSITION && velocity >= GOAL_VELOCITY;

        float reward = 0;
        if (done) {
            reward = 100;
        }
        reward -= Math.pow(actionData[0], 2) * 0.1;

        this.state[0] = position;
        this.state[1] = velocity;
        return new Snapshot(state, reward, done);
    }

    @Override
    public float[] reset() {
        episodeLength = 0;
        this.state[0] = random.nextFloat() * 0.2f - 0.6f;
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
