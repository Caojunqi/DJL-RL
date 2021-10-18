package algorithm.sac.block;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * Gaussian Policy
 *
 * @author Caojunqi
 * @date 2021-10-12 15:25
 */
public class GaussianPolicy extends MLP {
    private static final double LOG_SIG_MAX = 2;
    private static final double LOG_SIG_MIN = -6.907755;

    private int actionDim;

    public GaussianPolicy(int[] hiddenSizes, int actionDim) {
        // outputSize = actionDim * 2, Stack means and log_stds
        super(hiddenSizes, actionDim * 2);
        this.actionDim = actionDim;
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray meanAndLogStd = super.forwardInternal(parameterStore, inputs, training, params).singletonOrThrow();
        NDArray mean = meanAndLogStd.get(new NDIndex("..., :" + actionDim + ""));
        NDArray logStd = meanAndLogStd.get(new NDIndex("..., " + actionDim + ":"));
        NDArray std = logStd.clip(LOG_SIG_MIN, LOG_SIG_MAX).exp();
        return new NDList(mean, logStd, std);
    }
}
