package algorithm.sac.block;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * @author Caojunqi
 * @date 2021-10-26 11:40
 */
public class DiscreteGaussianPolicy extends MLP {

    private static final double LOG_SIG_MAX = 2;
    private static final double LOG_SIG_MIN = -6.907755;

    private int actionDim;

    public DiscreteGaussianPolicy(int[] hiddenSizes, int actionDim) {
        // outputSize = actionDim, Stack means
        super(hiddenSizes, actionDim);
        this.actionDim = actionDim;
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray distribution = super.forwardInternal(parameterStore, inputs, training, params).singletonOrThrow();
        distribution = distribution.softmax(distribution.getShape().dimension() - 1);
        return new NDList(distribution);
    }
}
