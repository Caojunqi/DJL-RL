package algorithm.ppo2;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import algorithm.BaseModelBlock;
import algorithm.ppo.PPOParameter;
import env.action.core.impl.DiscreteAction;
import utils.ActionSampler;
import utils.datatype.PolicyPair;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * @author Caojunqi
 * @date 2021-11-27 11:36
 */
public class DiscreteActorCriticModel extends BaseActorCriticModel<DiscreteAction> {

    private DiscreteActorCriticModel() {
        // 私有化构造器
    }

    public static DiscreteActorCriticModel newModel(NDManager manager, int stateDim, int actionNum) {
        Model model = Model.newInstance("discrete_actor_critic_model");
        BaseModelBlock net = new DiscreteActorCriticModelBlock(actionNum, PPOParameter.POLICY_MODEL_HIDDEN_SIZE, PPOParameter.CRITIC_MODEL_HIDDEN_SIZE);
        net.initialize(manager, DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);

        DiscreteActorCriticModel discreteActorCriticModel = new DiscreteActorCriticModel();
        discreteActorCriticModel.manager = manager;
        discreteActorCriticModel.model = model;
        discreteActorCriticModel.predictor = model.newPredictor(new NoopTranslator());
        return discreteActorCriticModel;
    }

    @Override
    public PolicyPair<DiscreteAction> policy(NDList states, boolean deterministic, boolean returnPolicyInfo, boolean noGrad) {
        try (NDManager subManager = manager.newSubManager()) {
            NDArray prob = predictor.predict(states).get(1);
            if (noGrad) {
                prob = prob.duplicate();
            }

            NDArray actionArray;
            if (deterministic) {
                actionArray = prob.argMax(-1).toType(DataType.INT32, false);
            } else {
                Random random = new Random(0);
                actionArray = ActionSampler.sampleMultinomial(subManager, prob, random);
            }

            int sampleSize = (int) actionArray.getShape().get(0);
            List<DiscreteAction> actions = new ArrayList<>();
            for (int i = 0; i < sampleSize; i++) {
                int actionData = actionArray.getInt(i);
                actions.add(new DiscreteAction(actionData));
            }
            return PolicyPair.of(actions, null);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public NDArray value(NDList states) {
        return null;
    }
}
