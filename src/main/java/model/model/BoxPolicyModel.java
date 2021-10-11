package model.model;

import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.NoopTranslator;
import env.common.action.impl.BoxAction;
import model.block.BaseModelBlock;
import model.block.BoxPolicyModelBlock;
import resource.ConstantParameter;

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
        BaseModelBlock net = new BoxPolicyModelBlock(actionDim, ConstantParameter.POLICY_MODEL_HIDDEN_SIZE, ConstantParameter.LOG_STD);
        net.initialize(manager, DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);

        BoxPolicyModel boxPolicyModel = new BoxPolicyModel();
        boxPolicyModel.manager = manager;
        boxPolicyModel.model = model;
        boxPolicyModel.predictor = model.newPredictor(new NoopTranslator());
        boxPolicyModel.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(ConstantParameter.LEARNING_RATE)).build();
        return boxPolicyModel;
    }
}
