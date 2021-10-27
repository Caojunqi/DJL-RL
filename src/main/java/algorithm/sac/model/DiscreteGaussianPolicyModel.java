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
import algorithm.sac.block.DiscreteGaussianPolicy;
import env.common.action.impl.DiscreteAction;
import utils.ActionSampler;
import utils.datatype.PolicyPair;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Caojunqi
 * @date 2021-10-26 11:41
 */
public class DiscreteGaussianPolicyModel extends BasePolicyModel<DiscreteAction> {

    private DiscreteGaussianPolicyModel() {
        // 私有化构造器
    }

    public static DiscreteGaussianPolicyModel newModel(NDManager manager, int stateDim, int actionDim) {
        Model model = Model.newInstance("gaussian_policy_model");
        BaseModelBlock net = new DiscreteGaussianPolicy(SACParameter.NETS_HIDDEN_SIZES, actionDim);
        net.initialize(manager, DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);

        DiscreteGaussianPolicyModel discreteGaussianPolicyModel = new DiscreteGaussianPolicyModel();
        discreteGaussianPolicyModel.manager = manager;
        discreteGaussianPolicyModel.model = model;
        discreteGaussianPolicyModel.predictor = model.newPredictor(new NoopTranslator());
        discreteGaussianPolicyModel.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(SACParameter.POLICY_LR)).build();
        return discreteGaussianPolicyModel;
    }

    @Override
    public PolicyPair<DiscreteAction> policy(NDList states, boolean deterministic, boolean returnPolicyInfo) {
        try (NDManager subManager = manager.newSubManager()) {
            NDArray distribution = predictor.predict(states).singletonOrThrow();
            NDArray actionArray;
            if (deterministic) {
                actionArray = distribution.argMax(-1).toType(DataType.INT32, false).duplicate();
            } else {
                actionArray = ActionSampler.sampleMultinomial(subManager, distribution, random);
            }
            int sampleSize = (int) actionArray.getShape().get(0);
            List<DiscreteAction> actions = new ArrayList<>();
            for (int i = 0; i < sampleSize; i++) {
                int actionData = actionArray.getInt(i);
                actions.add(new DiscreteAction(actionData));
            }

            NDList info = null;
            if (returnPolicyInfo) {
                // Have to deal with situation of 0.0 probabilities because we can't do log 0
                NDArray z = distribution.eq(0);
                z = z.toType(DataType.FLOAT32, false).mul(1e-8);
                NDArray logDistribution = distribution.add(z).log();
                info = new NDList(distribution, logDistribution);
            }

            return PolicyPair.of(actions, info);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }
}
