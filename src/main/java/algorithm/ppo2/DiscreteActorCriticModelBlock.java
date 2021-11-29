package algorithm.ppo2;

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
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.UniformInitializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.util.PairList;
import algorithm.BaseModelBlock;

/**
 * @author Caojunqi
 * @date 2021-11-27 11:35
 */
public class DiscreteActorCriticModelBlock extends BaseModelBlock {
    private int actionNum;
    // actor
    private int[] actorHiddenSize;
    private Block actorAffineLayer;
    private Block actionHead;
    // critic
    private int[] criticHiddenSize;
    private Block criticAffineLayer;
    private Block valueHead;

    public DiscreteActorCriticModelBlock(int actionNum, int[] actorHiddenSize, int[] criticHiddenSize) {
        super();
        this.actionNum = actionNum;
        // actor
        SequentialBlock actorAffineLayers = new SequentialBlock();
        for (int hiddenNum : actorHiddenSize) {
            actorAffineLayers.add(Linear.builder().setUnits(hiddenNum).build());
            actorAffineLayers.add(Activation::tanh);
        }
        this.actorHiddenSize = actorHiddenSize;
        this.actorAffineLayer = addChildBlock("actor_affine_layer", actorAffineLayers);
        this.actionHead = addChildBlock("action_head", Linear.builder().setUnits(actionNum).build());
        // critic
        SequentialBlock criticAffineLayers = new SequentialBlock();
        for (int hiddenNum : criticHiddenSize) {
            criticAffineLayers.add(Linear.builder().setUnits(hiddenNum).build());
            criticAffineLayers.add(Activation::tanh);
        }
        this.criticHiddenSize = criticHiddenSize;
        this.criticAffineLayer = addChildBlock("critic_affine_layer", criticAffineLayers);
        this.valueHead = addChildBlock("value_head", Linear.builder().setUnits(1).build());
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDList actorHidden = new NDList(actorAffineLayer.forward(parameterStore, inputs, training).singletonOrThrow());
        NDArray scores = actionHead.forward(parameterStore, actorHidden, training).singletonOrThrow();
        NDArray distribution = scores.softmax(scores.getShape().dimension() - 1);

        NDList criticHidden = new NDList(criticAffineLayer.forward(parameterStore, inputs, training).singletonOrThrow());
        NDArray value = valueHead.forward(parameterStore, criticHidden, training).singletonOrThrow();
        return new NDList(distribution, value);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{new Shape(actionNum), new Shape(1)};
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        // actor
        this.actorAffineLayer.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.actorAffineLayer.initialize(manager, dataType, inputShapes[0]);
        this.actionHead.setInitializer(new UniformInitializer(), Parameter.Type.WEIGHT);
        this.actionHead.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.actionHead.initialize(manager, dataType, new Shape(actorHiddenSize[actorHiddenSize.length - 1]));
        // critic
        this.criticAffineLayer.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.criticAffineLayer.initialize(manager, dataType, inputShapes[0]);
        this.valueHead.setInitializer(new UniformInitializer(), Parameter.Type.WEIGHT);
        this.valueHead.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.valueHead.initialize(manager, dataType, new Shape(criticHiddenSize[criticHiddenSize.length - 1]));
    }
}
