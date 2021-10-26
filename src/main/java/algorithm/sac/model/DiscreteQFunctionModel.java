package algorithm.sac.model;

import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.NoopTranslator;
import algorithm.BaseModelBlock;
import algorithm.ppo.model.BaseValueModel;
import algorithm.sac.SACParameter;
import algorithm.sac.block.DiscreteQFunction;

/**
 * @author Caojunqi
 * @date 2021-10-26 14:43
 */
public class DiscreteQFunctionModel extends BaseValueModel {

    private DiscreteQFunctionModel() {
        // 私有化构造器
    }

    public static DiscreteQFunctionModel newModel(NDManager manager, int stateDim, int actionDim) {
        Model model = Model.newInstance("q_function_model");
        BaseModelBlock net = new DiscreteQFunction(SACParameter.NETS_HIDDEN_SIZES, actionDim);
        net.initialize(manager, DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);

        DiscreteQFunctionModel discreteQFunctionModel = new DiscreteQFunctionModel();
        discreteQFunctionModel.manager = manager;
        discreteQFunctionModel.model = model;
        discreteQFunctionModel.predictor = model.newPredictor(new NoopTranslator());
        discreteQFunctionModel.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(SACParameter.QF_LR)).build();
        return discreteQFunctionModel;
    }
}
