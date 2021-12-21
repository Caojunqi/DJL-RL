package demo.mountaincar;

import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.ReplayBuffer;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import algorithm.ppo2.FixedBuffer;

import java.util.Random;

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
public class MountainCarContinuous implements RlEnv {
    private static final double[][] STATE_SPACE = new double[][]{{-1.2, 0.6}, {-0.07, 0.07}};
    private static final float[][] ACTION_SPACE = new float[][]{{-1.0f, 1.0f}};
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
    private static final double MAX_EPISODE_LENGTH = 200;

    private NDManager manager;
    private ReplayBuffer replayBuffer;
    private Random random;
    private State state;

    public MountainCarContinuous(NDManager manager, Random random, int batchSize, int bufferSize) {
        this.manager = manager;
        this.replayBuffer = new FixedBuffer(batchSize, bufferSize);
        this.random = random;
        this.state = new State(new float[]{0.0f, 0.0f}, 0);
    }

    @Override
    public void reset() {
        this.state.stateData[0] = random.nextFloat() * 0.2f - 0.6f;
        this.state.stateData[1] = 0;
        this.state.count = 0;
    }

    @Override
    public void close() {
        manager.close();
    }

    @Override
    public NDList getObservation() {
        return state.getObservation(manager);
    }

    @Override
    public ActionSpace getActionSpace() {
        return state.getActionSpace(manager);
    }

    @Override
    public Step step(NDList action, boolean training) {
        State preState = state;
        state = new State(preState.stateData.clone(), preState.count);
        float actionData = action.singletonOrThrow().getFloat();
        float position = this.state.stateData[0];
        float velocity = this.state.stateData[1];
        double force = Math.min(Math.max(actionData, MIN_ACTION), MAX_ACTION);
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
        reward -= Math.pow(actionData, 2) * 0.1;

        this.state.stateData[0] = position;
        this.state.stateData[1] = velocity;

        MountainCarContinuousStep step = new MountainCarContinuousStep(manager.newSubManager(), preState, state, action, reward, state.count++ > MAX_EPISODE_LENGTH || done);
        if (training) {
            replayBuffer.addStep(step);
        }
        return step;
    }

    @Override
    public Step[] getBatch() {
        return replayBuffer.getBatch();
    }

    static final class MountainCarContinuousStep implements RlEnv.Step {
        private NDManager manager;
        private State preState;
        private State postState;
        private NDList action;
        private float reward;
        private boolean done;

        private MountainCarContinuousStep(NDManager manager, State preState, State postState, NDList action, float reward, boolean done) {
            this.manager = manager;
            this.preState = preState;
            this.postState = postState;
            this.action = action;
            this.reward = reward;
            this.done = done;
        }

        @Override
        public NDList getPreObservation() {
            return preState.getObservation(manager);
        }

        @Override
        public NDList getAction() {
            return action;
        }

        @Override
        public NDList getPostObservation() {
            return postState.getObservation(manager);
        }

        @Override
        public ActionSpace getPostActionSpace() {
            return postState.getActionSpace(manager);
        }

        @Override
        public NDArray getReward() {
            return manager.create(reward);
        }

        @Override
        public boolean isDone() {
            return done;
        }

        @Override
        public void close() {
            manager.close();
        }
    }


    private static class State {
        float[] stateData;
        int count;

        private State(float[] stateData, int count) {
            this.stateData = stateData;
            this.count = count;
        }

        private NDList getObservation(NDManager manager) {
            return new NDList(manager.create(stateData));
        }

        private ActionSpace getActionSpace(NDManager manager) {
            ActionSpace actionSpace = new ActionSpace();
            actionSpace.add(new NDList(manager.create(ACTION_SPACE)));
            return actionSpace;
        }
    }
}
