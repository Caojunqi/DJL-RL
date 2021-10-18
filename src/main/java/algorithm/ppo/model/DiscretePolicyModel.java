package algorithm.ppo.model;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import algorithm.BaseModelBlock;
import algorithm.CommonParameter;
import algorithm.ppo.PPOParameter;
import algorithm.ppo.block.DiscretePolicyModelBlock;
import env.common.action.impl.DiscreteAction;
import utils.ActionSampler;
import utils.datatype.PolicyPair;

/**
 * 离散型动作策略模型
 *
 * @author Caojunqi
 * @date 2021-09-23 22:03
 */
public class DiscretePolicyModel extends BasePolicyModel<DiscreteAction> {

    private DiscretePolicyModel() {
        // 私有化构造器
    }

    public static DiscretePolicyModel newModel(NDManager manager, int stateDim, int actionNum) {
        Model model = Model.newInstance("discrete_policy_model");
        BaseModelBlock net = new DiscretePolicyModelBlock(actionNum, PPOParameter.POLICY_MODEL_HIDDEN_SIZE);
        net.initialize(manager, DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);

        DiscretePolicyModel discretePolicyModel = new DiscretePolicyModel();
        discretePolicyModel.manager = manager;
        discretePolicyModel.model = model;
        discretePolicyModel.predictor = model.newPredictor(new NoopTranslator());
        discretePolicyModel.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(CommonParameter.LEARNING_RATE)).build();
        return discretePolicyModel;
    }

    @Override
    public PolicyPair<DiscreteAction> policy(NDList states, boolean deterministic, boolean returnLogProb) {
        try {
            NDArray prob = predictor.predict(states).singletonOrThrow();
            int actionData;
            if (deterministic) {
                actionData = ActionSampler.greedy(prob);
            } else {
                actionData = ActionSampler.sampleMultinomial(prob, random);
            }
            DiscreteAction action = new DiscreteAction(actionData);
            return PolicyPair.of(action, null);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }
}
