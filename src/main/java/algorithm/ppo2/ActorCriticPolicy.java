package algorithm.ppo2;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.PairList;
import env.action.core.IAction;
import env.action.space.IActionSpace;
import env.state.core.IState;
import env.state.space.IStateSpace;
import utils.ActionSampler;

import java.util.List;
import java.util.Random;

/**
 * @author Caojunqi
 * @date 2021-11-29 10:39
 */
public class ActorCriticPolicy<S extends IState<S>, A extends IAction> extends AbstractBlock {
    private NDManager manager = NDManager.newBaseManager();
    private Random random = new Random(0);
    // BaseModel
    private IStateSpace<S> observationSpace;
    private IActionSpace<A> actionSpace;
    private boolean normalizeImages;
    // BasePolicy
    private boolean squashOutput;
    // ActorCriticPolicy
    private boolean orthoInit;
    private float logStdInit;
    private boolean useSde;

    private MlpExtractor mlpExtractor;
    private Block actionNet;
    private Block valueNet;
    private Optimizer optimizer;

    public ActorCriticPolicy(IStateSpace<S> observationSpace,
                             IActionSpace<A> actionSpace,
                             String lrSchedule,
                             boolean orthoInit,
                             boolean useSde,
                             float logStdInit,
                             boolean fullStd,
                             List<Integer> sdeNetArch,
                             boolean useExpln,
                             boolean squashOutput,
                             boolean normalizeImages) {
        this.observationSpace = observationSpace;
        this.actionSpace = actionSpace;
        this.normalizeImages = normalizeImages;

        this.squashOutput = squashOutput;

        this.orthoInit = orthoInit;
        this.logStdInit = logStdInit;
        this.useSde = useSde;

        int[] shared = new int[]{};
        int[] policy = new int[]{64, 64};
        int[] value = new int[]{64, 64};
        this.mlpExtractor = addChildBlock("mlp_extractor", new MlpExtractor(shared, policy, value, Activation::tanh));

        int actionDim = actionSpace.getDim();
        this.actionNet = addChildBlock("action_net", Linear.builder().setUnits(actionDim).build());

        this.valueNet = addChildBlock("value_net", Linear.builder().setUnits(1).build());

        float learningRate = 0.01f;
        this.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(learningRate)).build();
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDList flatten = new NDList(inputs.get(0).reshape(inputs.get(0).getShape().get(0), -1));
        NDList mlp = this.mlpExtractor.forward(parameterStore, flatten, training);
        NDArray values = this.valueNet.forward(parameterStore, new NDList(mlp.get(1)), training).singletonOrThrow();
        NDArray meanAction = this.actionNet.forward(parameterStore, new NDList(mlp.get(0)), training).singletonOrThrow();
        NDArray actionProb = meanAction.softmax(-1);
        NDArray actions = ActionSampler.sampleMultinomial(manager, actionProb, random);
        NDArray actionLogProb = actionProb.log();
        return new NDList(actions, values, actionLogProb);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[0];
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        this.mlpExtractor.initialize(manager, dataType, inputShapes[0]);

        this.actionNet.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.actionNet.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.actionNet.initialize(manager, dataType, new Shape(this.mlpExtractor.getPolicyOutputSize()));

        this.valueNet.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.valueNet.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.valueNet.initialize(manager, dataType, new Shape(this.mlpExtractor.getValueOutputSize()));
    }
}
