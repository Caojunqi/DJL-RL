package ppo;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import algorithm.ppo.PPODiscrete;
import algorithm.ppo.model.BasePolicyModel;
import algorithm.ppo.model.BaseValueModel;
import algorithm.ppo.model.CriticValueModel;
import algorithm.ppo.model.DiscretePolicyModel;
import demo.cartpole.CartPole;
import env.action.core.impl.DiscreteAction;
import env.state.core.impl.BoxState;
import utils.Runner;

/**
 * PPO算法测试类
 *
 * @author Caojunqi
 * @date 2021-09-09 20:57
 */
public class PPODiscreteTest {

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(0);
        CartPole env = new CartPole(false);
        NDManager manager = NDManager.newBaseManager();
        BasePolicyModel<DiscreteAction> policyModel = DiscretePolicyModel.newModel(manager, env.getStateSpaceDim(), env.getActionSpaceDim());
        BaseValueModel valueModel = CriticValueModel.newModel(manager, env.getStateSpaceDim());
        PPODiscrete<BoxState> algorithm = new PPODiscrete<>(manager, policyModel, valueModel);
        env.seed(0);
        new Runner<>(env, algorithm)
                .mainLoop();
    }
}
