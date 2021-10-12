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
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.util.PairList;
import algorithm.BaseModelBlock;

/**
 * 用于生成连续型动作的策略模型
 *
 * @author Caojunqi
 * @date 2021-09-16 18:13
 */
public class BoxPolicyModelBlock extends BaseModelBlock {
    private final String ACTION_LOG_STD_PARAMETER = "action_log_std";
    private int actionDim;
    private int[] hiddenSize;
    private float logStd;
    private Block affineLayer;
    private Block actionMean;
    private Parameter actionLogStd;

    public BoxPolicyModelBlock(int actionDim, int[] hiddenSize, float logStd) {
        super();
        SequentialBlock affineLayers = new SequentialBlock();
        for (int hiddenNum : hiddenSize) {
            affineLayers.add(Linear.builder().setUnits(hiddenNum).build());
            affineLayers.add(Activation::tanh);
        }
        this.actionDim = actionDim;
        this.hiddenSize = hiddenSize;
        this.logStd = logStd;
        this.affineLayer = addChildBlock("affine_layer", affineLayers);
        this.actionMean = addChildBlock("action_mean", Linear.builder().setUnits(actionDim).build());
        this.actionLogStd = addParameter(Parameter.builder().
                setType(Parameter.Type.OTHER).
                setName(ACTION_LOG_STD_PARAMETER).
                optInitializer(new ActionLogStdInitializer()).
                optRequiresGrad(true).
                optShape(new Shape(1, actionDim)).build());
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDList hidden = new NDList(affineLayer.forward(parameterStore, inputs, training).singletonOrThrow());
        NDArray mean = actionMean.forward(parameterStore, hidden, training).singletonOrThrow();
        NDArray logStd = actionLogStd.getArray().broadcast(mean.getShape()).duplicate();
        NDArray std = logStd.exp();
        return new NDList(mean, logStd, std);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{new Shape(actionDim)};
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.affineLayer.initialize(manager, dataType, inputShapes[0]);
        this.actionMean.initialize(manager, dataType, new Shape(hiddenSize[hiddenSize.length - 1]));
    }

    class ActionLogStdInitializer implements Initializer {

        @Override
        public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
            return manager.ones(shape, dataType, manager.getDevice()).mul(logStd);
        }
    }
}
