package model.one;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.util.PairList;
import model.BaseModel;

/**
 * @author Caojunqi
 * @date 2021-09-10 14:35
 */
public class DistributionValueModel extends BaseModel {
    private static final float LAYERNORM_MOMENTUM = 0.9999f;
    private static final float LAYERNORM_EPSILON = 1e-5f;
    private Block linearInput;
    private Block linearAction;
    private Block linearValue;

    private int hiddenSize;
    private int outputSize;
    private Parameter gamma;
    private Parameter beta;
    private float movingMean = 0.0f;
    private float movingVar = 1.0f;

    private DistributionValueModel(NDManager manager, int hiddenSize, int outputSize) {
        super(manager);
        this.linearInput = addChildBlock("linear_input", Linear.builder().setUnits(hiddenSize).build());
        this.linearAction = addChildBlock("linear_action", Linear.builder().setUnits(outputSize).build());
        this.linearValue = addChildBlock("linear_value", Linear.builder().setUnits(1).build());

        this.gamma = addParameter(Parameter.builder().setName("mu").setType(Parameter.Type.GAMMA).optRequiresGrad(true).optShape(new Shape(1)).build());
        this.beta = addParameter(Parameter.builder().setName("sigma").setType(Parameter.Type.BETA).optRequiresGrad(true).optShape(new Shape(1)).build());

        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
    }

    public static Model newModel(NDManager manager, int inputSize, int hiddenSize, int outputSize) {
        Model model = Model.newInstance("distribution_value_model");
        BaseModel net = new DistributionValueModel(manager, hiddenSize, outputSize);
        net.initialize(net.getManager(), DataType.FLOAT32, new Shape(inputSize));
        model.setBlock(net);

        return model;
    }

    @Override
    public NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training,
                                  PairList<String, Object> params) {

        NDList hidden = new NDList(
                Activation.relu(linearInput.forward(parameterStore, inputs, training).singletonOrThrow()));
        NDArray scores = normalize(linearAction.forward(parameterStore, hidden, training).singletonOrThrow());
        NDArray distribution = scores.softmax(scores.getShape().dimension() - 1);

        NDArray value = linearValue.forward(parameterStore, hidden, training).singletonOrThrow();

        return new NDList(distribution, value);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{new Shape(outputSize), new Shape(1)};
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        linearInput.initialize(manager, dataType, inputShapes[0]);
        linearAction.initialize(manager, dataType, new Shape(hiddenSize));
        linearValue.initialize(manager, dataType, new Shape(hiddenSize));
    }

    private NDArray normalize(NDArray arr) {
        float score_mean = arr.mean().getFloat();
        movingMean = movingMean * LAYERNORM_MOMENTUM + score_mean * (1.0f - LAYERNORM_MOMENTUM);
        movingVar = movingVar * LAYERNORM_MOMENTUM
                + arr.sub(score_mean).pow(2).mean().getFloat() * (1.0f - LAYERNORM_MOMENTUM);
        return arr.sub(movingMean).div(Math.sqrt(movingVar + LAYERNORM_EPSILON)).mul(gamma.getArray())
                .add(beta.getArray());
    }
}
