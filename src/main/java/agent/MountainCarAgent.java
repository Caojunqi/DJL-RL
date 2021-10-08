package agent;

import algorithm.AlgorithmType;
import env.common.action.impl.DiscreteAction;
import env.demo.mountaincar.MountainCar;
import model.model.CriticValueModel;
import model.model.DiscretePolicyModel;

/**
 * 针对MountainCar的Agent
 *
 * @author Caojunqi
 * @date 2021-09-15 11:09
 */
public class MountainCarAgent extends BaseAgent<DiscreteAction, MountainCar> {

    public MountainCarAgent(MountainCar env, AlgorithmType algorithmType) {
        super(env, algorithmType);
        policyModel = DiscretePolicyModel.newModel(manager, env.getStateSpaceDim(), env.getActionSpaceDim());
        valueModel = CriticValueModel.newModel(manager, env.getStateSpaceDim());
    }

    @Override
    public void collect(float[] state, DiscreteAction action, boolean done, float[] nextState, float reward) {
        memory.addTransition(state, action, done, nextState, reward);
    }
}
