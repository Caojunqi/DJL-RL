package model.one;

import ai.djl.Model;
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
public class ScoreModel extends BaseModel {
    private Block linearInput;
    private Block linearOutput;

    private int hiddenSize;
    private int outputSize;

    protected ScoreModel(NDManager manager, int hiddenSize, int outputSize) {
        super(manager);
        this.linearInput = addChildBlock("linear_input", Linear.builder().setUnits(hiddenSize).build());
        this.linearOutput = addChildBlock("linear_output", Linear.builder().setUnits(outputSize).build());

        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
    }

    public static Model newModel(NDManager manager, int inputSize, int hiddenSize, int outputSize) {
        Model model = Model.newInstance("score_model");
        BaseModel net = new ScoreModel(manager, hiddenSize, outputSize);
        net.initialize(net.getManager(), DataType.FLOAT32, new Shape(inputSize));
        model.setBlock(net);

        return model;
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDList hidden = new NDList(
                Activation.relu(linearInput.forward(parameterStore, inputs, training).singletonOrThrow()));
        return linearOutput.forward(parameterStore, hidden, training);

    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{new Shape(outputSize)};
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        linearInput.initialize(manager, dataType, inputShapes[0]);
        linearOutput.initialize(manager, dataType, new Shape(hiddenSize));
    }
}
