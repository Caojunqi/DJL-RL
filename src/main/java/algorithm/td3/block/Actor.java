package algorithm.td3.block;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import algorithm.sac.block.MLP;

/**
 * TD3 Actor
 *
 * @author Caojunqi
 * @date 2021-11-01 11:42
 */
public class Actor extends MLP {

    private int actionDim;

    public Actor(int[] hiddenSizes, int actionDim) {
        super(hiddenSizes, actionDim);
        this.actionDim = actionDim;
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray mean = super.forwardInternal(parameterStore, inputs, training, params).singletonOrThrow();
        return new NDList(mean);
    }
}
