package algorithm.sac.model;

import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoopTranslator;
import algorithm.BaseModelBlock;
import algorithm.ppo.model.BaseValueModel;
import algorithm.sac.SACParameter;
import algorithm.sac.block.QFunction;

/**
 * Q函数模型
 *
 * @author Caojunqi
 * @date 2021-10-12 17:37
 */
public class QFunctionModel extends BaseValueModel {

    private QFunctionModel() {
        // 私有化构造器
    }

    public static QFunctionModel newModel(NDManager manager, int stateDim, int actionDim) {
        Model model = Model.newInstance("q_function_model");
        BaseModelBlock net = new QFunction(SACParameter.NETS_HIDDEN_SIZES);
        net.initialize(manager, DataType.FLOAT32, new Shape(stateDim + actionDim));
        model.setBlock(net);

        QFunctionModel qFunctionModel = new QFunctionModel();
        qFunctionModel.manager = manager;
        qFunctionModel.model = model;
        qFunctionModel.predictor = model.newPredictor(new NoopTranslator());
        return qFunctionModel;
    }
}
