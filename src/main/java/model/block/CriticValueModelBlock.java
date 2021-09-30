package model.block;

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

/**
 * 状态价值函数近似模型
 *
 * @author Caojunqi
 * @date 2021-09-16 18:06
 */
public class CriticValueModelBlock extends BaseModelBlock {
    private int[] hiddenSize;
    private Block affineLayer;
    private Block valueHead;

    public CriticValueModelBlock(int[] hiddenSize) {
        super();
        SequentialBlock affineLayers = new SequentialBlock();
        for (int hiddenNum : hiddenSize) {
            affineLayers.add(Linear.builder().setUnits(hiddenNum).build());
            affineLayers.add(Activation::tanh);
        }
        this.hiddenSize = hiddenSize;
        this.affineLayer = addChildBlock("affine_layer", affineLayers);
        this.valueHead = addChildBlock("value_head", Linear.builder().setUnits(1).build());
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDList hidden = new NDList(affineLayer.forward(parameterStore, inputs, training).singletonOrThrow());
        NDArray value = valueHead.forward(parameterStore, hidden, training).singletonOrThrow();
        return new NDList(value);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{new Shape(1)};
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        this.affineLayer.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.affineLayer.initialize(manager, dataType, inputShapes[0]);

        this.valueHead.setInitializer(new UniformInitializer(), Parameter.Type.WEIGHT);
        this.valueHead.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.valueHead.initialize(manager, dataType, new Shape(hiddenSize[hiddenSize.length - 1]));
    }
}
