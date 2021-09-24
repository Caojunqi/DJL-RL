package model.model;

import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import model.block.BaseModelBlock;
import model.block.BoxPolicyModelBlock;

/**
 * 连续型动作策略模型
 *
 * @author Caojunqi
 * @date 2021-09-23 22:01
 */
public class BoxPolicyModel extends BasePolicyModel {


    public static Model newModel(NDManager manager, int stateDim, int actionDim) {
        int[] hiddenSize = new int[]{128, 128};
        return newModel(manager, stateDim, actionDim, hiddenSize, 0.0f);
    }

    public static Model newModel(NDManager manager, int stateDim, int actionDim, int[] hiddenSize) {
        return newModel(manager, stateDim, actionDim, hiddenSize, 0.0f);
    }

    public static Model newModel(NDManager manager, int stateDim, int actionDim, float logStd) {
        int[] hiddenSize = new int[]{128, 128};
        return newModel(manager, stateDim, actionDim, hiddenSize, logStd);
    }

    public static Model newModel(NDManager manager, int stateDim, int actionDim, int[] hiddenSize, float logStd) {
        Model model = Model.newInstance("box_policy_model");
        BaseModelBlock net = new BoxPolicyModelBlock(actionDim, hiddenSize, logStd);
        net.initialize(manager, DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);
        return model;
    }
}
