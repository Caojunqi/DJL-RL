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
public class DiscreteActorModel extends BaseModel {
    private int actionNum;
    private Block actor;

    private DiscreteActorModel(NDManager manager, int actionNum) {
        super(manager);

        this.actionNum = actionNum;
        SequentialBlock actorSequentialBlock = new SequentialBlock();
        actorSequentialBlock.add(Linear.builder().setUnits(64).build());
        actorSequentialBlock.add(Activation::tanh);
        actorSequentialBlock.add(Linear.builder().setUnits(64).build());
        actorSequentialBlock.add(Activation::tanh);
        actorSequentialBlock.add(Linear.builder().setUnits(actionNum).build());
        // todo 缺softmax
        this.actor = actorSequentialBlock;
    }

    public static Model newModel(NDManager manager, int stateDim, int actionNum) {
        Model model = Model.newInstance("discrete_actor_model");
        BaseModel net = new DiscreteActorModel(manager, actionNum);
        net.initialize(net.getManager(), DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);
        return model;
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray scores = this.actor.forward(parameterStore, inputs, training).singletonOrThrow();
        NDArray distribution = scores.softmax(scores.getShape().dimension() - 1);
        return new NDList(distribution);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{new Shape(actionNum)};
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.actor.initialize(manager, dataType, inputShapes[0]);
    }
}
