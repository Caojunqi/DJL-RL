package demo.cartpole;

import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.ReplayBuffer;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import algorithm.ppo2.FixedBuffer;

import java.util.Random;

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
public class CartPole implements RlEnv {
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

    private NDManager manager;
    private ReplayBuffer replayBuffer;
    private Random random;
    private State state;

    public CartPole(NDManager manager, Random random, int batchSize, int bufferSize) {
        this.manager = manager;
        this.replayBuffer = new FixedBuffer(batchSize, bufferSize);
        this.random = random;
        this.state = new State(new float[]{0.0f, 0.0f, 0.0f, 0.0f}, 0);
    }

    @Override
    public void reset() {
        for (int i = 0; i < 4; i++) {
            state.stateData[i] = random.nextFloat() * 0.1f - 0.05f;
        }
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
        double force = actionData == 1 ? FORCE_MAG : -FORCE_MAG;
        double cos_theta = Math.cos(state.stateData[2]);
        double sin_theta = Math.sin(state.stateData[2]);
        double temp = (force + POLEMASS_LENGTH * Math.pow(state.stateData[3], 2) * sin_theta) / TOTAL_MASS;

        double theta_acc = ((GRAVITY * sin_theta - temp * cos_theta)
                / (LENGTH * (4.0 / 3.0 - POLE_MASS * Math.pow(cos_theta, 2) / TOTAL_MASS)));
        double x_acc = temp - POLEMASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

        state.stateData[0] += TAU * state.stateData[1];
        state.stateData[1] += TAU * x_acc;
        state.stateData[2] += TAU * state.stateData[3];
        if (state.stateData[2] > Math.PI) {
            state.stateData[2] -= 2 * Math.PI;
        } else if (state.stateData[2] < -Math.PI) {
            state.stateData[2] += 2 * Math.PI;
        }
        state.stateData[3] += TAU * theta_acc;
        boolean done = (state.stateData[0] < -X_THRESHOLD || state.stateData[0] > X_THRESHOLD || state.stateData[2] < -THETA_THRESHOLD
                || state.stateData[2] > THETA_THRESHOLD);

        CartPoleStep step = new CartPoleStep(manager.newSubManager(), preState, state, action, 1.0f, state.count++ > MAX_EPISODE_LENGTH || done);
        if (training) {
            replayBuffer.addStep(step);
        }
        return step;
    }

    @Override
    public Step[] getBatch() {
        return replayBuffer.getBatch();
    }

    static final class CartPoleStep implements RlEnv.Step {
        private NDManager manager;
        private State preState;
        private State postState;
        private NDList action;
        private float reward;
        private boolean done;

        private CartPoleStep(NDManager manager, State preState, State postState, NDList action, float reward, boolean done) {
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
            for (int i = 0; i < 2; i++) {
                actionSpace.add(new NDList(manager.create(i)));
            }
            return actionSpace;
        }
    }

}
