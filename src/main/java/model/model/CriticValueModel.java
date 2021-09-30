package model.model;

import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.NoopTranslator;
import model.block.BaseModelBlock;
import model.block.CriticValueModelBlock;
import resource.ConstantParameter;

/**
 * 状态价值函数近似模型
 *
 * @author Caojunqi
 * @date 2021-09-23 22:02
 */
public class CriticValueModel extends BaseValueModel {

    private CriticValueModel() {
        // 私有化构造器
    }

    public static CriticValueModel newModel(NDManager manager, int stateDim) {
        Model model = Model.newInstance("critic_value_model");
        BaseModelBlock net = new CriticValueModelBlock(ConstantParameter.CRITIC_MODEL_HIDDEN_SIZE);
        net.initialize(manager, DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);

        CriticValueModel criticValueModel = new CriticValueModel();
        criticValueModel.manager = manager;
        criticValueModel.model = model;
        criticValueModel.predictor = model.newPredictor(new NoopTranslator());
        criticValueModel.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(ConstantParameter.LEARNING_RATE)).build();
        return criticValueModel;
    }
}
