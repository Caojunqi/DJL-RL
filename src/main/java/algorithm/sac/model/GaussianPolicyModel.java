package algorithm.sac.model;

import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.NoopTranslator;
import algorithm.BaseModelBlock;
import algorithm.ppo.model.BasePolicyModel;
import algorithm.sac.SACParameter;
import algorithm.sac.block.GaussianPolicy;
import env.common.action.impl.BoxAction;

/**
 * Gaussian策略模型
 *
 * @author Caojunqi
 * @date 2021-10-12 17:09
 */
public class GaussianPolicyModel extends BasePolicyModel<BoxAction> {

    private GaussianPolicyModel() {
        // 私有化构造器
    }

    public static GaussianPolicyModel newModel(NDManager manager, int stateDim, int actionDim) {
        Model model = Model.newInstance("gaussian_policy_model");
        BaseModelBlock net = new GaussianPolicy(SACParameter.NETS_HIDDEN_SIZES, actionDim);
        net.initialize(manager, DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);

        GaussianPolicyModel gaussianPolicyModel = new GaussianPolicyModel();
        gaussianPolicyModel.manager = manager;
        gaussianPolicyModel.model = model;
        gaussianPolicyModel.predictor = model.newPredictor(new NoopTranslator());
        gaussianPolicyModel.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(SACParameter.POLICY_LR)).build();
        return gaussianPolicyModel;
    }
}
