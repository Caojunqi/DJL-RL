package algorithm.td3.model;

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
import algorithm.ppo.model.BasePolicyModel;
import algorithm.td3.TD3Parameter;
import algorithm.td3.block.Actor;
import env.common.action.impl.BoxAction;
import utils.datatype.PolicyPair;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Caojunqi
 * @date 2021-11-01 11:43
 */
public class ActorModel extends BasePolicyModel<BoxAction> {

    private ActorModel() {
        // 私有化构造器
    }

    public static ActorModel newModel(NDManager manager, int stateDim, int actionDim) {
        Model model = Model.newInstance("actor_model");
        BaseModelBlock net = new Actor(TD3Parameter.NETS_HIDDEN_SIZES, actionDim);
        net.initialize(manager, DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);

        ActorModel actorModel = new ActorModel();
        actorModel.manager = manager;
        actorModel.model = model;
        actorModel.predictor = model.newPredictor(new NoopTranslator());
        return actorModel;
    }

    @Override
    public PolicyPair<BoxAction> policy(NDList states, boolean deterministic, boolean returnPolicyInfo, boolean noGrad) {
        try (NDManager subManager = manager.newSubManager()) {
            NDList distribution = predictor.predict(states);
            NDArray mean = distribution.singletonOrThrow();
            if (noGrad) {
                mean = mean.duplicate();
            }

            NDArray action;
            if (deterministic) {
                action = mean;
            } else {
                action = normalSampleActionArray(subManager, mean);
            }

            int actionSize = (int) action.getShape().get(0);
            List<BoxAction> actions = new ArrayList<>();
            for (int i = 0; i < actionSize; i++) {
                float[] actionData = action.get(new NDIndex(i + ",:")).toFloatArray();
                actions.add(new BoxAction(actionData));
            }

            NDList info = null;
            if (returnPolicyInfo) {
                info = new NDList(action);
            }

            return PolicyPair.of(actions, info);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    public NDArray normalSampleActionArray(NDManager manager, NDArray actionMean) {
        NDArray noise = manager.randomNormal(0f, TD3Parameter.ACTION_NOISE_STD, actionMean.getShape(), actionMean.getDataType());
        noise = noise.clip(-TD3Parameter.ACTION_NOISE_CLIPPING_RANGE, TD3Parameter.ACTION_NOISE_CLIPPING_RANGE);
        return actionMean.add(noise);
    }
}
