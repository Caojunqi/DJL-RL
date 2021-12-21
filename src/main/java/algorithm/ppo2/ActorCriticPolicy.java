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
import ai.djl.util.PairList;
import algorithm.ppo.PPOParameter;
import utils.ActionSampler;

import java.util.Random;

/**
 * @author Caojunqi
 * @date 2021-11-29 10:39
 */
public class ActorCriticPolicy extends AbstractBlock {
    private NDManager mainManager;
    private Random random;

    private MlpExtractor mlpExtractor;
    private Block actionNet;
    private Block valueNet;

    public ActorCriticPolicy(NDManager mainManager, Random random, int actionDim) {
        this.mainManager = mainManager;
        this.random = random;

        int[] shared = new int[]{};
        this.mlpExtractor = addChildBlock("mlp_extractor", new MlpExtractor(shared, PPOParameter.POLICY_MODEL_HIDDEN_SIZE, PPOParameter.CRITIC_MODEL_HIDDEN_SIZE, Activation::tanh));

        this.actionNet = addChildBlock("action_net", Linear.builder().setUnits(actionDim).build());

        this.valueNet = addChildBlock("value_net", Linear.builder().setUnits(1).build());
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDList flatten = new NDList(inputs.get(0).reshape(inputs.get(0).getShape().get(0), -1));
        NDList mlp = this.mlpExtractor.forward(parameterStore, flatten, training);
        NDArray values = this.valueNet.forward(parameterStore, new NDList(mlp.get(1)), training).singletonOrThrow();
        NDArray meanAction = this.actionNet.forward(parameterStore, new NDList(mlp.get(0)), training).singletonOrThrow();
        // Normalize
        meanAction = meanAction.sub(meanAction.exp().sum(new int[]{-1}, true).log());
//        NDArray mask = meanAction.getManager().create(new float[]{0, 1}).toType(DataType.BOOLEAN, false).broadcast(meanAction.getShape());
//        NDArray maskPolicy = NDArrays.where(mask, meanAction, meanAction.getManager().create(-1e8f));
//        maskPolicy = maskPolicy.sub(maskPolicy.exp().sum(new int[]{-1},true).log());
        NDArray actionProb = meanAction.softmax(-1);
        NDArray actions;
        if (training) {
            actions = ActionSampler.sampleMultinomial(mainManager, actionProb, random);
        } else {
            actions = actionProb.argMax().toType(DataType.INT32, false);
            actions.attach(mainManager);
        }
        NDArray entropy = meanAction.mul(actionProb).sum(new int[]{-1}).neg();
        return new NDList(actions, values, meanAction, entropy);
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
