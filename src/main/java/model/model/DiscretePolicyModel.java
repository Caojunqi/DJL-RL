package model.model;

import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.NoopTranslator;
import env.common.action.impl.DiscreteAction;
import model.block.BaseModelBlock;
import model.block.DiscretePolicyModelBlock;
import resource.ConstantParameter;

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
        BaseModelBlock net = new DiscretePolicyModelBlock(actionNum, ConstantParameter.POLICY_MODEL_HIDDEN_SIZE);
        net.initialize(manager, DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);

        DiscretePolicyModel discretePolicyModel = new DiscretePolicyModel();
        discretePolicyModel.manager = manager;
        discretePolicyModel.model = model;
        discretePolicyModel.predictor = model.newPredictor(new NoopTranslator());
        discretePolicyModel.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(ConstantParameter.LEARNING_RATE)).build();
        return discretePolicyModel;
    }
}
