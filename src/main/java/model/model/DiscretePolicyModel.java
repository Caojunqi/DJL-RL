package model.model;

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
import env.common.action.impl.DiscreteAction;
import model.block.BaseModelBlock;
import model.block.DiscretePolicyModelBlock;
import resource.ConstantParameter;
import utils.ActionSampler;

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

    @Override
    public DiscreteAction selectAction(float[] state) {
        // 此处将单一状态数组转为多维的，这样可以保证在predict过程中，传入1个状态和传入多个状态，输入数据的维度是一致的。
        float[][] states = new float[][]{state};
        try (NDManager subManager = manager.newSubManager()) {
            NDArray prob = predictor.predict(new NDList(subManager.create(states))).singletonOrThrow();
            int actionData = ActionSampler.sampleMultinomial(prob, random);
            return new DiscreteAction(actionData);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public DiscreteAction greedyAction(float[] state) {
        // 此处将单一状态数组转为多维的，这样可以保证在predict过程中，传入1个状态和传入多个状态，输入数据的维度是一致的。
        float[][] states = new float[][]{state};
        try (NDManager subManager = manager.newSubManager()) {
            NDArray prob = predictor.predict(new NDList(subManager.create(states))).singletonOrThrow();
            int actionData = ActionSampler.greedy(prob);
            return new DiscreteAction(actionData);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }
}
