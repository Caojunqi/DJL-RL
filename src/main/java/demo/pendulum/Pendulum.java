package demo.pendulum;

import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.ReplayBuffer;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.util.RandomUtils;
import algorithm.ppo2.FixedBuffer;

import java.util.Random;

/**
 * 复刻gym中的Pendulum-v1环境
 *
 * @author Caojunqi
 * @date 2021-11-02 17:12
 */
public class Pendulum implements RlEnv {

    private static final double MAX_SPEED = 8;
    private static final float MAX_TORQUE = 2.0f;
    private static final double DT = 0.05;
    private static final double G = 10.0;
    private static final double M = 1.0;
    private static final double L = 1.0;
    private static final double[][] STATE_SPACE = new double[][]{{-1.0, 1.0}, {-1.0, 1.0}, {-MAX_SPEED, MAX_SPEED}};
    private static final float[][] ACTION_SPACE = new float[][]{{-MAX_TORQUE, MAX_TORQUE}};
    private static final double MAX_EPISODE_LENGTH = 200;

    private NDManager manager;
    private ReplayBuffer replayBuffer;
    private Random random;
    private State state;

    public Pendulum(NDManager manager, Random random, int batchSize, int bufferSize) {
        this.manager = manager;
        this.replayBuffer = new FixedBuffer(batchSize, bufferSize);
        this.random = random;
        this.state = new State(new float[]{0.0f, 0.0f}, 0);
    }

    @Override
    public void reset() {
        this.state.stateData[0] = RandomUtils.nextFloat((float) -Math.PI, (float) Math.PI);
        this.state.stateData[1] = RandomUtils.nextFloat(-1, 1);
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

        float th = this.state.stateData[0];
        float thdot = this.state.stateData[1];
        double u = Math.min(actionData, MAX_TORQUE);
        u = Math.max(u, -MAX_TORQUE);

        double costs = Math.pow(angleNormalize(th), 2) + 0.1 * Math.pow(thdot, 2) + 0.001 * Math.pow(u, 2);
        double newThdot = thdot + (3 * G / (2 * L) * Math.sin(th) + 3.0 / (M * Math.pow(L, 2)) * u) * DT;
        newThdot = Math.min(newThdot, MAX_SPEED);
        newThdot = Math.max(newThdot, -MAX_SPEED);
        double newTh = th + newThdot * DT;

        this.state.stateData[0] = (float) newTh;
        this.state.stateData[1] = (float) newThdot;

        PendulumStep step = new PendulumStep(manager.newSubManager(), preState, state, action, (float) -costs, state.count++ > MAX_EPISODE_LENGTH);
        if (training) {
            replayBuffer.addStep(step);
        }
        return step;
    }

    @Override
    public Step[] getBatch() {
        return replayBuffer.getBatch();
    }

    private double angleNormalize(float x) {
        return ((x + Math.PI) % (2 * Math.PI)) - Math.PI;
    }

    static final class PendulumStep implements RlEnv.Step {
        private NDManager manager;
        private State preState;
        private State postState;
        private NDList action;
        private float reward;
        private boolean done;

        private PendulumStep(NDManager manager, State preState, State postState, NDList action, float reward, boolean done) {
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
