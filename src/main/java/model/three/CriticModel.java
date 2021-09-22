package model.three;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.util.PairList;
import model.BaseModel;

/**
 * @author Caojunqi
 * @date 2021-09-18 10:52
 */
public class CriticModel extends BaseModel {

    private Block critic;

    private CriticModel(NDManager manager) {
        super(manager);
        SequentialBlock criticSequentialBlock = new SequentialBlock();
        criticSequentialBlock.add(Linear.builder().setUnits(64).build());
        criticSequentialBlock.add(Activation::tanh);
        criticSequentialBlock.add(Linear.builder().setUnits(64).build());
        criticSequentialBlock.add(Activation::tanh);
        criticSequentialBlock.add(Linear.builder().setUnits(1).build());
        this.critic = criticSequentialBlock;
    }

    public static Model newModel(NDManager manager, int stateDim) {
        Model model = Model.newInstance("critic_model");
        BaseModel net = new CriticModel(manager);
        net.initialize(net.getManager(), DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);
        return model;
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray value = this.critic.forward(parameterStore, inputs, training).singletonOrThrow();
        return new NDList(value);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{new Shape(1)};
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.critic.initialize(manager, dataType, inputShapes[0]);
    }
}
