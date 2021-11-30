package algorithm.ppo2;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.util.PairList;

import java.util.function.Function;

/*_
 *  Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
 *  a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
 *  of them are shared between the policy network and the value network. It is assumed to be a list with the following
 *  structure:
 *
 *  1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
 *     If the number of ints is zero, there will be no shared layers.
 *  2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
 *     It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
 *     If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.
 *
 *  For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
 *  network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
 *  would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
 *  would be specified as [128, 128].
 *
 *  Adapted from Stable Baselines3.
 *
 * @author Caojunqi
 * @date 2021-11-29 11:10
 */
public class MlpExtractor extends AbstractBlock {
    private Block sharedLayer;
    private int sharedOutputSize;
    private Block policyLayer;
    private int policyOutputSize;
    private Block valueLayer;
    private int valueOutputSize;

    /**
     * @param shared policy和value共享的网络
     * @param policy policy独享的网络
     * @param value  value独享的网络
     */
    public MlpExtractor(int[] shared, int[] policy, int[] value, Function<NDList, NDList> activation) {
        SequentialBlock sharedLayer = new SequentialBlock();
        int sharedOutputSize = 0;
        for (int netNum : shared) {
            sharedLayer.add(Linear.builder().setUnits(netNum).build());
            sharedLayer.add(activation);
            sharedOutputSize = netNum;
        }
        this.sharedOutputSize = sharedOutputSize;
        this.sharedLayer = addChildBlock("shared_layer", sharedLayer);

        SequentialBlock policyLayer = new SequentialBlock();
        int policyOutputSize = 0;
        for (int netNum : policy) {
            policyLayer.add(Linear.builder().setUnits(netNum).build());
            policyLayer.add(activation);
            policyOutputSize = netNum;
        }
        this.policyOutputSize = policyOutputSize;
        this.policyLayer = addChildBlock("policy_layer", policyLayer);

        SequentialBlock valueLayer = new SequentialBlock();
        int valueOutputSize = 0;
        for (int netNum : value) {
            valueLayer.add(Linear.builder().setUnits(netNum).build());
            valueLayer.add(activation);
            valueOutputSize = netNum;
        }
        this.valueOutputSize = valueOutputSize;
        this.valueLayer = addChildBlock("value_layer", valueLayer);
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDList shared = new NDList(sharedLayer.forward(parameterStore, inputs, training).singletonOrThrow());
        NDArray policy = policyLayer.forward(parameterStore, shared, training).singletonOrThrow();
        NDArray value = valueLayer.forward(parameterStore, shared, training).singletonOrThrow();
        return new NDList(policy, value);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{new Shape(policyOutputSize), new Shape(valueOutputSize)};
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        this.sharedLayer.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.sharedLayer.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.sharedLayer.initialize(manager, dataType, inputShapes[0]);

        this.policyLayer.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.policyLayer.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.policyLayer.initialize(manager, dataType, new Shape(sharedOutputSize));

        this.valueLayer.setInitializer(new XavierInitializer(), Parameter.Type.WEIGHT);
        this.valueLayer.setInitializer(Initializer.ZEROS, Parameter.Type.BIAS);
        this.valueLayer.initialize(manager, dataType, new Shape(sharedOutputSize));
    }

    public NDList forwardActor(ParameterStore parameterStore, NDList inputs, boolean training) {
        NDList shared = new NDList(sharedLayer.forward(parameterStore, inputs, training).singletonOrThrow());
        NDArray policy = policyLayer.forward(parameterStore, shared, training).singletonOrThrow();
        return new NDList(policy);
    }

    public NDList forwardCritic(ParameterStore parameterStore, NDList inputs, boolean training) {
        NDList shared = new NDList(sharedLayer.forward(parameterStore, inputs, training).singletonOrThrow());
        NDArray value = valueLayer.forward(parameterStore, shared, training).singletonOrThrow();
        return new NDList(value);
    }

    public int getPolicyOutputSize() {
        return policyOutputSize;
    }

    public int getValueOutputSize() {
        return valueOutputSize;
    }
}
