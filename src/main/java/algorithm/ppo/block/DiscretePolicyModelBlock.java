package algorithm.ppo.block;

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
 * 用于生成离散型动作的策略模型
 *
 * @author Caojunqi
 * @date 2021-09-16 15:40
 */
public class DiscretePolicyModelBlock extends BaseModelBlock {
    private int actionNum;
    private int[] hiddenSize;
    private Block affineLayer;
    private Block actionHead;

    public DiscretePolicyModelBlock(int actionNum, int[] hiddenSize) {
        super();
        SequentialBlock affineLayers = new SequentialBlock();
        for (int hiddenNum : hiddenSize) {
            affineLayers.add(Linear.builder().setUnits(hiddenNum).build());
            affineLayers.add(Activation::tanh);
        }
        this.actionNum = actionNum;
        this.hiddenSize = hiddenSize;
        this.affineLayer = addChildBlock("affine_layer", affineLayers);
        this.actionHead = addChildBlock("action_head", Linear.builder().setUnits(actionNum).build());
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDList hidden = new NDList(affineLayer.forward(parameterStore, inputs, training).singletonOrThrow());
        NDArray scores = actionHead.forward(parameterStore, hidden, training).singletonOrThrow();
        NDArray distribution = scores.softmax(scores.getShape().dimension() - 1);
        return new NDList(distribution);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{new Shape(actionNum)};
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        this.affineLayer.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.affineLayer.initialize(manager, dataType, inputShapes[0]);

        this.actionHead.setInitializer(new UniformInitializer(), Parameter.Type.WEIGHT);
        this.actionHead.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.actionHead.initialize(manager, dataType, new Shape(hiddenSize[hiddenSize.length - 1]));
    }

}
