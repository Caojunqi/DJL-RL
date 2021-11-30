package demo.pendulum;

import ai.djl.util.RandomUtils;
import env.action.core.impl.BoxAction;
import env.action.space.impl.BoxActionSpace;
import env.common.Environment;
import env.state.core.impl.BoxState;
import env.state.space.impl.BoxStateSpace;
import utils.datatype.Snapshot;

/**
 * 复刻gym中的Pendulum-v1环境
 *
 * @author Caojunqi
 * @date 2021-11-02 17:12
 */
public class Pendulum extends Environment<BoxState, BoxAction> {

    private static final double MAX_SPEED = 8;
    private static final double MAX_TORQUE = 2.0;
    private static final double DT = 0.05;
    private static final double G = 10.0;
    private static final double M = 1.0;
    private static final double L = 1.0;
    private static final double[][] STATE_SPACE = new double[][]{{-1.0, 1.0}, {-1.0, 1.0}, {-MAX_SPEED, MAX_SPEED}};
    private static final double[][] ACTION_SPACE = new double[][]{{-MAX_TORQUE, MAX_TORQUE}};
    private static final double MAX_EPISODE_LENGTH = 200;

    private final float[] state = new float[]{0.0f, 0.0f};

    private int episodeLength = 0;

    public Pendulum() {
        super(new BoxStateSpace(STATE_SPACE), new BoxActionSpace(ACTION_SPACE));
    }

    @Override
    protected Snapshot<BoxState> doStep(BoxAction action) {
        float th = this.state[0];
        float thdot = this.state[1];
        float[] actionData = action.getActionData();
        double u = Math.min(actionData[0], MAX_TORQUE);
        u = Math.max(u, -MAX_TORQUE);

        double costs = Math.pow(angleNormalize(th), 2) + 0.1 * Math.pow(thdot, 2) + 0.001 * Math.pow(u, 2);
        double newThdot = thdot + (3 * G / (2 * L) * Math.sin(th) + 3.0 / (M * Math.pow(L, 2)) * u) * DT;
        newThdot = Math.min(newThdot, MAX_SPEED);
        newThdot = Math.max(newThdot, -MAX_SPEED);
        double newTh = th + newThdot * DT;

        this.state[0] = (float) newTh;
        this.state[1] = (float) newThdot;
        return new Snapshot<>(new BoxState(getState()), (float) -costs, episodeLength++ > MAX_EPISODE_LENGTH);
    }

    @Override
    public BoxState reset() {
        this.state[0] = RandomUtils.nextFloat((float) -Math.PI, (float) Math.PI);
        this.state[1] = RandomUtils.nextFloat(-1, 1);
        return new BoxState(getState());
    }

    @Override
    public void render() {
        // do nothing!!
    }

    @Override
    public void close() {

    }

    /**
     * 获取环境当前的状态空间参数
     */
    private float[] getState() {
        float[] stateSpace = new float[3];
        stateSpace[0] = (float) Math.cos(this.state[0]);
        stateSpace[1] = (float) Math.sin(this.state[0]);
        stateSpace[2] = this.state[1];
        return stateSpace;
    }

    private double angleNormalize(float x) {
        return ((x + Math.PI) % (2 * Math.PI)) - Math.PI;
    }
}
