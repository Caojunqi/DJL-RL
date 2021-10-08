package agent;

import algorithm.AlgorithmType;
import env.common.action.impl.BoxAction;
import env.demo.mountaincar.MountainCarContinuous;
import model.model.BoxPolicyModel;
import model.model.CriticValueModel;

/**
 * 针对MountainCarContinuous的Agent
 *
 * @author Caojunqi
 * @date 2021-09-23 11:57
 */
public class MountainCarContinuousAgent extends BaseAgent<BoxAction, MountainCarContinuous> {

    /**
     * GAE参数
     */
    private float gaeLambda;
    private float gamma;
    private float l2Reg = 0.001f;

    /**
     * PPO参数
     */
    private int innerUpdates;
    private int innerBatchSize;
    private float ratioLowerBound;
    private float ratioUpperBound;

    public MountainCarContinuousAgent(MountainCarContinuous env, AlgorithmType algorithmType) {
        super(env, algorithmType);
        policyModel = BoxPolicyModel.newModel(manager, env.getStateSpaceDim(), env.getActionSpaceDim());
        valueModel = CriticValueModel.newModel(manager, env.getStateSpaceDim());
    }

    @Override
    public void collect(float[] state, BoxAction action, boolean done, float[] nextState, float reward) {
        memory.addTransition(state, action, done, nextState, reward);
    }
}
