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
public class MountainCar implements RlEnv {
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

    private NDManager manager;
    private ReplayBuffer replayBuffer;
    private Random random;
    private State state;

    public MountainCar(NDManager manager, Random random, int batchSize, int bufferSize) {
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
        int actionData = action.singletonOrThrow().getInt();
        state.stateData[1] += (actionData - 1) * FORCE - Math.cos(3 * state.stateData[0]) * (-GRAVITY);
        state.stateData[1] = Math.min(Math.max(state.stateData[1], -MAX_SPEED), MAX_SPEED);
        state.stateData[0] += state.stateData[1];
        state.stateData[0] = Math.min(Math.max(state.stateData[0], MIN_POSITION), MAX_POSITION);

        if (state.stateData[0] == MIN_POSITION && state.stateData[1] < 0) {
            state.stateData[1] = 0;
        }
        boolean done = state.stateData[0] >= GOAL_POSITION && state.stateData[1] >= GOAL_VELOCITY;

        MountainCarStep step = new MountainCarStep(manager.newSubManager(), preState, state, action, -1.0f, state.count++ > MAX_EPISODE_LENGTH || done);
        if (training) {
            replayBuffer.addStep(step);
        }
        return step;
    }

    @Override
    public Step[] getBatch() {
        return replayBuffer.getBatch();
    }

    static final class MountainCarStep implements RlEnv.Step {
        private NDManager manager;
        private State preState;
        private State postState;
        private NDList action;
        private float reward;
        private boolean done;

        private MountainCarStep(NDManager manager, State preState, State postState, NDList action, float reward, boolean done) {
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
            for (int i = 0; i < 3; i++) {
                actionSpace.add(new NDList(manager.create(i)));
            }
            return actionSpace;
        }
    }
}
