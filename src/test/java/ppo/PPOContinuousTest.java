package ppo;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import algorithm.ppo.PPOContinuous;
import algorithm.ppo.model.BasePolicyModel;
import algorithm.ppo.model.BaseValueModel;
import algorithm.ppo.model.BoxPolicyModel;
import algorithm.ppo.model.CriticValueModel;
import demo.mountaincar.MountainCarContinuous;
import env.action.core.impl.BoxAction;
import env.state.core.impl.BoxState;
import utils.Runner;

/**
 * PPO算法测试类
 *
 * @author Caojunqi
 * @date 2021-09-09 20:57
 */
public class PPOContinuousTest {

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(0);
        NDManager manager = NDManager.newBaseManager();
        MountainCarContinuous env = new MountainCarContinuous(false);
        BasePolicyModel<BoxAction> policyModel = BoxPolicyModel.newModel(manager, env.getStateSpaceDim(), env.getActionSpaceDim());
        BaseValueModel valueModel = CriticValueModel.newModel(manager, env.getStateSpaceDim());
        PPOContinuous<BoxState> algorithm = new PPOContinuous<>(manager, policyModel, valueModel);
        env.seed(0);
        new Runner<>(env, algorithm)
                .mainLoop();
    }
}
