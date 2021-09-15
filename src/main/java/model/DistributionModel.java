package model;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * @author Caojunqi
 * @date 2021-09-10 14:35
 */
public class DistributionModel extends ScoreModel {
    private DistributionModel(NDManager manager, int hiddenSize, int outputSize) {
        super(manager, hiddenSize, outputSize);
    }

    public static Model newModel(NDManager manager, int inputSize, int hiddenSize, int outputSize) {
        Model model = Model.newInstance("distribution_model");
        BaseModel net = new DistributionModel(manager, hiddenSize, outputSize);
        net.initialize(net.getManager(), DataType.FLOAT32, new Shape(inputSize));
        model.setBlock(net);

        return model;
    }

    @Override
    public NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training,
                                  PairList<String, Object> params) {
        NDArray scores = super.forward(parameterStore, inputs, training, params).singletonOrThrow();

        return new NDList(scores.softmax(scores.getShape().dimension() - 1));
    }
}
