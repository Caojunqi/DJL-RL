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
import algorithm.sac.block.VFunction;

/**
 * 状态函数模型
 *
 * @author Caojunqi
 * @date 2021-10-12 17:44
 */
public class VFunctionModel extends BaseValueModel {

    private VFunctionModel() {
        // 私有化构造器
    }

    public static VFunctionModel newModel(NDManager manager, int stateDim) {
        Model model = Model.newInstance("v_function_model");
        BaseModelBlock net = new VFunction(SACParameter.NETS_HIDDEN_SIZES);
        net.initialize(manager, DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);

        VFunctionModel vFunctionModel = new VFunctionModel();
        vFunctionModel.manager = manager;
        vFunctionModel.model = model;
        vFunctionModel.predictor = model.newPredictor(new NoopTranslator());
        vFunctionModel.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(SACParameter.QF_LR)).build();
        return vFunctionModel;
    }
}
