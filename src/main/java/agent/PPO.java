package agent;

import agent.base.BaseAgent;
import agent.model.DistributionValueModel;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import utils.ActionSampler;
import utils.Memory;

import java.util.Random;

/**
 * PPO算法核心
 *
 * @author Caojunqi
 * @date 2021-09-09 21:10
 */
public class PPO extends BaseAgent {
    protected Random random = new Random(0);
    protected Memory memory = new Memory(1024);
    protected Optimizer optimizer;

    protected NDManager manager = NDManager.newBaseManager();
    protected Model model;
    protected Predictor<NDList, NDList> predictor;

    /**
     * 环境参数
     */
    private int actionNum;
    private int stateSpaceDim;
    private int hiddenSize;

    /**
     * GAE参数
     */
    private float gaeLambda;
    private float gamma;

    /**
     * PPO参数
     */
    private int innerUpdates;
    private int innerBatchSize;
    private float ratioLowerBound;
    private float ratioUpperBound;

    public PPO(int stateSpaceDim, int actionNum, int hiddenSize,
               float gamma, float gaeLambda,
               float learningRate, int innerUpdates, int innerBatchSize, float ratioClip) {
        this.stateSpaceDim = stateSpaceDim;
        this.actionNum = actionNum;
        this.hiddenSize = hiddenSize;
        this.gamma = gamma;
        this.gaeLambda = gaeLambda;
        this.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(learningRate)).build();
        if (manager != null) {
            manager.close();
        }
        manager = NDManager.newBaseManager();
        model = DistributionValueModel.newModel(manager, stateSpaceDim, hiddenSize, actionNum);
        predictor = model.newPredictor(new NoopTranslator());
        this.innerUpdates = innerUpdates;
        this.innerBatchSize = innerBatchSize;
        this.ratioLowerBound = 1.0f - ratioClip;
        this.ratioUpperBound = 1.0f + ratioClip;
    }

    @Override
    public int react(float[] state) {
        try (NDManager subManager = manager.newSubManager()) {
            NDArray prob = predictor.predict(new NDList(subManager.create(state))).get(0);
            return ActionSampler.sampleMultinomial(prob, random);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void collect(float reward, boolean done) {

    }

    protected void updateModel(NDManager submanager) throws TranslateException {
        MemoryBatch batch = memory.getOrderedBatch(submanager);
        NDArray states = batch.getStates();
        NDArray actions = batch.getActions();

        NDList net_output = predictor.predict(new NDList(states));

        NDArray distribution = Helper.gather(net_output.get(0).duplicate(), actions.toIntArray());
        NDArray values = net_output.get(1).duplicate();

        NDList estimates = estimateAdvantage(values.duplicate(), batch.getRewards());
        NDArray expected_returns = estimates.get(0);
        NDArray advantages = estimates.get(1);

        int[] index = new int[inner_batch_size];

        for (int i = 0; i < inner_updates * (1 + batch.size() / inner_batch_size); i++) {
            for (int j = 0; j < inner_batch_size; j++) {
                index[j] = random.nextInt(batch.size());
            }
            NDArray states_subset = getSample(submanager, states, index);
            NDArray actions_subset = getSample(submanager, actions, index);
            NDArray distribution_subset = getSample(submanager, distribution, index);
            NDArray expected_returns_subset = getSample(submanager, expected_returns, index);
            NDArray advantages_subset = getSample(submanager, advantages, index);

            NDList net_output_updated = predictor.predict(new NDList(states_subset));
            NDArray distribution_updated = Helper.gather(net_output_updated.get(0), actions_subset.toIntArray());
            NDArray values_updated = net_output_updated.get(1);

            NDArray loss_critic = (expected_returns_subset.sub(values_updated)).square().sum();

            NDArray ratios = distribution_updated.div(distribution_subset).expandDims(1);

            NDArray loss_actor = ratios.clip(ratio_lower_bound, ratio_upper_bound).mul(advantages_subset)
                    .minimum(ratios.mul(advantages_subset)).sum().neg();
            NDArray loss = loss_actor.add(loss_critic);

            try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                collector.backward(loss);

                for (Pair<String, Parameter> params : model.getBlock().getParameters()) {
                    NDArray params_arr = params.getValue().getArray();

                    optimizer.update(params.getKey(), params_arr, params_arr.getGradient().duplicate());
                }

            }
        }
    }

    private NDArray getSample(NDManager submanager, NDArray array, int[] index) {

        Shape shape = Shape.update(array.getShape(), 0, inner_batch_size);
        NDArray sample = submanager.zeros(shape, array.getDataType());
        for (int i = 0; i < inner_batch_size; i++) {
            sample.set(new NDIndex(i), array.get(index[i]));
        }
        return sample;
    }
}
