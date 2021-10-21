package algorithm.sac.model;

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
import algorithm.ppo.model.BasePolicyModel;
import algorithm.sac.SACParameter;
import algorithm.sac.block.GaussianPolicy;
import env.common.action.impl.BoxAction;
import utils.ActionSampler;
import utils.datatype.PolicyPair;

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

    @Override
    public PolicyPair<BoxAction> policy(NDList states, boolean deterministic, boolean returnLogProb) {
        try {
            NDList distribution = predictor.predict(states);
            NDArray mean = distribution.get(0).duplicate();
            NDArray logStd = distribution.get(1).duplicate();
            NDArray std = distribution.get(2).duplicate();
            double[] actionData;
            if (deterministic) {
                actionData = ActionSampler.sampleNormalGreedy(mean);
            } else {
                actionData = ActionSampler.sampleNormal(mean, std, random);
            }
            BoxAction action = new BoxAction(actionData);

            NDList info = null;
            if (returnLogProb) {
                NDManager subManager = manager.newSubManager();
                NDArray logProb = ActionSampler.sampleLogProb(subManager.create(actionData), mean, std, logStd);
                info = new NDList(logProb);
            }
            return PolicyPair.of(action, info);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }
}
