package env.common;

import env.common.spaces.Space;
import utils.datatype.Snapshot;

import java.util.Random;

/*_
 * The main environment class. It encapsulates an environment with
 * arbitrary behind-the-scenes dynamics. An environment can be
 * partially or fully observed.
 *
 * The main API methods that users of this class need to know are:
 *
 *     step
 *     reset
 *     render
 *     close
 *     seed
 *
 * And set the following attributes:
 *
 *     actionSpace: The Space object corresponding to valid actions
 *     stateSpace: The Space object corresponding to valid states
 *     rewardRange: A tuple corresponding to the min and max possible rewards
 *
 * Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
 *
 * The methods are accessed publicly as "step", "reset", etc...
 *
 * @author Caojunqi
 * @date 2021-09-09 20:59
 */
public abstract class Environment {
    /**
     * 随机数生成器
     */
    protected Random random = new Random(0);

    /**
     * 状态空间
     */
    private Space stateSpace;
    /**
     * 动作空间
     */
    private Space actionSpace;
    /**
     * 收益取值范围
     */
    private double[] rewardRange;

    public Environment(Space stateSpace, Space actionSpace) {
        this(stateSpace, actionSpace, new double[]{Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY});
    }

    public Environment(Space stateSpace, Space actionSpace, double[] rewardRange) {
        this.stateSpace = stateSpace;
        this.actionSpace = actionSpace;
        this.rewardRange = rewardRange;
    }

    /*_
     * Run one timestep of the environment's dynamics. When end of
     * episode is reached, you are responsible for calling `reset()`
     * to reset this environment's state.
     *
     * @param action an action provided by the agent
     * @return environment snapshot after the action. Contains the following information:
     *          state (object): agent's observation of the current environment
     *          reward (float) : amount of reward returned after previous action
     *          done (bool): whether the episode has ended, in which case further step() calls will return undefined results
     */
    public abstract Snapshot step(int action);

    /**
     * Resets the environment to an initial state and returns an initial
     * state.
     * <p>
     * Note that this function should not reset the environment's random
     * number generator({@link this#random}); random variables in the environment's state should
     * be sampled independently between multiple calls to `reset()`. In other
     * words, each call of `reset()` should yield an environment suitable for
     * a new episode, independent of previous episodes.
     *
     * @return state (object): the initial state.
     */
    public abstract float[] reset();

    /**
     * Renders the environment.
     */
    public abstract void render();

    /**
     * Override close in your subclass to perform any necessary cleanup.
     */
    public abstract void close();

    /*_
     * Sets the seed for this env's random number generator.
     *
     * Note:
     *     Some environments use multiple pseudorandom number generators.
     *     We want to capture all such seeds used in order to ensure that
     *     there aren't accidental correlations between multiple generators.
     */
    public void seed(long seed) {
        random.setSeed(seed);
    }

}
