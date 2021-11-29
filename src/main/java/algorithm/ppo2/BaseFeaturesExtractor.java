package algorithm.ppo2;

import ai.djl.nn.AbstractBlock;
import env.state.core.IState;
import env.state.space.IStateSpace;

/**
 * @author Caojunqi
 * @date 2021-11-29 15:18
 */
public abstract class BaseFeaturesExtractor<S extends IState<S>> extends AbstractBlock {

    private IStateSpace<S> stateSpace;

    public BaseFeaturesExtractor(IStateSpace<S> stateSpace) {
        this.stateSpace = stateSpace;
    }

    public int getFeaturesDim() {
        return stateSpace.getFlatDim();
    }
}
