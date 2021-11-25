package algorithm.ppo.model;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import algorithm.BaseModelBlock;
import algorithm.CommonParameter;
import algorithm.ppo.PPOParameter;
import algorithm.ppo.block.BoxPolicyModelBlock;
import env.action.core.impl.BoxAction;
import utils.datatype.PolicyPair;

import java.util.ArrayList;
import java.util.List;

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
        return boxPolicyModel;
    }

    @Override
    public PolicyPair<BoxAction> policy(NDList states, boolean deterministic, boolean returnPolicyInfo, boolean noGrad) {
        try (NDManager subManager = manager.newSubManager()) {
            NDList distribution = predictor.predict(states);
            NDArray mean = distribution.get(0);
            NDArray logStd = distribution.get(1);
            NDArray std = distribution.get(2);
            if (noGrad) {
                mean = mean.duplicate();
                logStd = logStd.duplicate();
                std = std.duplicate();
            }

            NDArray actionArray;
            if (deterministic) {
                actionArray = mean;
            } else {
                NDArray noise = subManager.randomNormal(std.getShape());
                actionArray = std.mul(noise).add(mean);
            }

            int actionSize = (int) actionArray.getShape().get(0);
            List<BoxAction> actions = new ArrayList<>();
            for (int i = 0; i < actionSize; i++) {
                float[] actionData = actionArray.get(new NDIndex(i + ",:")).toFloatArray();
                actions.add(new BoxAction(actionData));
            }
            return PolicyPair.of(actions, null);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }
}
