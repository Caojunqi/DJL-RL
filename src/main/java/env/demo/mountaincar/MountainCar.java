package env.demo.mountaincar;

import env.common.Environment;
import env.common.spaces.Box;
import env.common.spaces.Discrete;
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
public class MountainCar extends Environment {
    private static final double[][] STATE_SPACE = new double[][]{{-1.2, 0.6}, {-0.07, 0.07}};
    private static final float MIN_POSITION = -1.2f;
    private static final float MAX_POSITION = 0.6f;
    private static final float MAX_SPEED = 0.1f;
    private static final float GOAL_POSITION = 0.5f;
    private static final float GOAL_VELOCITY = 0.0f;
    private static final double FORCE = 0.001;
    private static final double GRAVITY = 0.0025;
    private static final double MAX_EPISODE_LENGTH = 200;

    private final float[] state = new float[]{0.0f, 0.0f};

    private int episodeLength = 0;

    public MountainCar() {
        super(new Box(STATE_SPACE), new Discrete(3));
    }

    @Override
    public Snapshot step(int action) {
        return null;
    }

    @Override
    public float[] reset() {
        return null;
    }

    @Override
    public void render() {
    }

    @Override
    public void close() {

    }
}
