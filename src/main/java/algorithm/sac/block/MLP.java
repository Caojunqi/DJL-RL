package algorithm.sac.block;

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
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.util.PairList;
import algorithm.BaseModelBlock;

/**
 * Multilayer Perceptron for modeling policies, Q-values and state values.
 *
 * @author Caojunqi
 * @date 2021-10-12 15:23
 */
public abstract class MLP extends BaseModelBlock {
    protected int[] hiddenSizes;
    protected int outputSize;
    protected Block affineLayer;
    protected Block outputLayer;

    public MLP(int[] hiddenSizes, int outputSize) {
        super();
        SequentialBlock affineLayers = new SequentialBlock();
        for (int hiddenNum : hiddenSizes) {
            affineLayers.add(Linear.builder().setUnits(hiddenNum).build());
            affineLayers.add(Activation::relu);
        }
        this.hiddenSizes = hiddenSizes;
        this.outputSize = outputSize;
        this.affineLayer = addChildBlock("affine_layer", affineLayers);
        this.outputLayer = addChildBlock("output_layer", Linear.builder().setUnits(outputSize).build());
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDList hidden = new NDList(affineLayer.forward(parameterStore, inputs, training).singletonOrThrow());
        NDArray scores = outputLayer.forward(parameterStore, hidden, training).singletonOrThrow();
        return new NDList(scores);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{new Shape(outputSize)};
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        this.affineLayer.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.affineLayer.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.affineLayer.initialize(manager, dataType, inputShapes[0]);

        this.outputLayer.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.outputLayer.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.outputLayer.initialize(manager, dataType, new Shape(hiddenSizes[hiddenSizes.length - 1]));
    }
}
