package model.three;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import model.BaseModel;

/**
 * @author Caojunqi
 * @date 2021-09-17 21:20
 */
public class ActorCriticModel extends BaseModel {
    private boolean hasContinuousActionSpace;
    private int actionDim;
    private NDArray actionVar;
    private Block actor;
    private Block critic;

    public ActorCriticModel(NDManager manager, int stateDim, int actionDim, boolean hasContinuousActionSpace, float actionStdInit) {
        super(manager);
        this.hasContinuousActionSpace = hasContinuousActionSpace;
        if (hasContinuousActionSpace) {
            this.actionDim = actionDim;
            this.actionVar = manager.full(new Shape(actionDim), actionStdInit * actionStdInit, DataType.FLOAT32, manager.getDevice());
        }

        // actor
        if (hasContinuousActionSpace) {
            SequentialBlock actorSequentialBlock = new SequentialBlock();
            actorSequentialBlock.add(Linear.builder().setUnits(64).build());
            actorSequentialBlock.add(Activation::tanh);
            actorSequentialBlock.add(Linear.builder().setUnits(64).build());
            actorSequentialBlock.add(Activation::tanh);
            actorSequentialBlock.add(Linear.builder().setUnits(actionDim).build());
            actorSequentialBlock.add(Activation::tanh);
            this.actor = actorSequentialBlock;
        } else {
            SequentialBlock actorSequentialBlock = new SequentialBlock();
            actorSequentialBlock.add(Linear.builder().setUnits(64).build());
            actorSequentialBlock.add(Activation::tanh);
            actorSequentialBlock.add(Linear.builder().setUnits(64).build());
            actorSequentialBlock.add(Activation::tanh);
            actorSequentialBlock.add(Linear.builder().setUnits(actionDim).build());
            // todo 缺softmax
            this.actor = actorSequentialBlock;
        }

        // critic
        SequentialBlock criticSequentialBlock = new SequentialBlock();
        criticSequentialBlock.add(Linear.builder().setUnits(64).build());
        criticSequentialBlock.add(Activation::tanh);
        criticSequentialBlock.add(Linear.builder().setUnits(64).build());
        criticSequentialBlock.add(Activation::tanh);
        criticSequentialBlock.add(Linear.builder().setUnits(1).build());
        this.critic = criticSequentialBlock;
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        return null;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[0];
    }
}
