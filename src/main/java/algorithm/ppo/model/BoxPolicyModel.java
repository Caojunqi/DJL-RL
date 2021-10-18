package algorithm.ppo.model;

import ai.djl.Model;
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
import algorithm.ppo.block.BoxPolicyModelBlock;
import env.common.action.impl.BoxAction;
import utils.ActionSampler;
import utils.datatype.PolicyPair;

/**
 * 连续型动作策略模型
 *
 * @author Caojunqi
 * @date 2021-09-23 22:01
 */
public class BoxPolicyModel extends BasePolicyModel<BoxAction> {

    private BoxPolicyModel() {
        // 私有化构造器
    }

    public static BoxPolicyModel newModel(NDManager manager, int stateDim, int actionDim) {
        Model model = Model.newInstance("box_policy_model");
        BaseModelBlock net = new BoxPolicyModelBlock(actionDim, PPOParameter.POLICY_MODEL_HIDDEN_SIZE, CommonParameter.LOG_STD);
        net.initialize(manager, DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);

        BoxPolicyModel boxPolicyModel = new BoxPolicyModel();
        boxPolicyModel.manager = manager;
        boxPolicyModel.model = model;
        boxPolicyModel.predictor = model.newPredictor(new NoopTranslator());
        boxPolicyModel.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(CommonParameter.LEARNING_RATE)).build();
        return boxPolicyModel;
    }

    @Override
    public PolicyPair<BoxAction> policy(NDList states, boolean deterministic, boolean returnLogProb) {
        try {
            NDList distribution = predictor.predict(states);
            double[] actionData;
            if (deterministic) {
                actionData = ActionSampler.sampleNormalGreedy(distribution.get(0));
            } else {
                actionData = ActionSampler.sampleNormal(distribution.get(0), distribution.get(2), random);
            }
            BoxAction action = new BoxAction(actionData);
            return PolicyPair.of(action, null);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }
}
