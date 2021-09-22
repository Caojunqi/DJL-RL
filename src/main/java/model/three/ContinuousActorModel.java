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
public class ContinuousActorModel extends BaseModel {
    private int actionDim;
    private NDArray actionVar;
    private Block actor;

    private ContinuousActorModel(NDManager manager, int actionDim, float actionStdInit) {
        super(manager);
        this.actionDim = actionDim;
        this.actionVar = manager.full(new Shape(actionDim), actionStdInit * actionStdInit, DataType.FLOAT32, manager.getDevice());

        SequentialBlock actorSequentialBlock = new SequentialBlock();
        actorSequentialBlock.add(Linear.builder().setUnits(64).build());
        actorSequentialBlock.add(Activation::tanh);
        actorSequentialBlock.add(Linear.builder().setUnits(64).build());
        actorSequentialBlock.add(Activation::tanh);
        actorSequentialBlock.add(Linear.builder().setUnits(actionDim).build());
        actorSequentialBlock.add(Activation::tanh);
        this.actor = actorSequentialBlock;
    }

    public static Model newModel(NDManager manager, int stateDim, int actionDim, float actionStdInit) {
        Model model = Model.newInstance("continuous_actor_model");
        BaseModel net = new ContinuousActorModel(manager, actionDim, actionStdInit);
        net.initialize(net.getManager(), DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);
        return model;
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray actionMean = this.actor.forward(parameterStore, inputs, training).singletonOrThrow();
        return new NDList(actionMean);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{new Shape(actionDim)};
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.actor.initialize(manager, dataType, inputShapes[0]);
    }
}
