package agent;

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
import env.common.action.impl.DiscreteAction;
import env.demo.mountaincar.MountainCar;
import model.DistributionValueModel;
import utils.ActionSampler;
import utils.Helper;
import utils.Memory;
import utils.MemoryBatch;

import java.util.Random;

/**
 * 针对MountainCar的Agent
 *
 * @author Caojunqi
 * @date 2021-09-15 11:09
 */
public class MountainCarAgent extends BaseAgent<DiscreteAction, MountainCar> {

    protected Random random = new Random(0);
    protected Memory<DiscreteAction> memory = new Memory<>();
    protected Optimizer optimizer;

    protected NDManager manager = NDManager.newBaseManager();
    protected Model model;
    protected Predictor<NDList, NDList> predictor;

    /**
     * GAE参数
     */
    private float gaeLambda;
    private float gamma;

    /**
     * PPO参数
     */
    private int hiddenSize;
    private int innerUpdates;
    private int innerBatchSize;
    private float ratioLowerBound;
    private float ratioUpperBound;

    public MountainCarAgent(MountainCar env, int hiddenSize,
                            float gamma, float gaeLambda,
                            float learningRate, int innerUpdates, int innerBatchSize, float ratioClip) {
        super(env);
        this.hiddenSize = hiddenSize;
        this.gamma = gamma;
        this.gaeLambda = gaeLambda;
        this.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(learningRate)).build();
        if (manager != null) {
            manager.close();
        }
        manager = NDManager.newBaseManager();
        model = DistributionValueModel.newModel(manager, env.getStateSpaceDim(), hiddenSize, env.getActionSpaceDim());
        predictor = model.newPredictor(new NoopTranslator());
        this.innerUpdates = innerUpdates;
        this.innerBatchSize = innerBatchSize;
        this.ratioLowerBound = 1.0f - ratioClip;
        this.ratioUpperBound = 1.0f + ratioClip;
    }

    @Override
    public DiscreteAction selectAction(float[] state) {
        try (NDManager subManager = manager.newSubManager()) {
            NDArray prob = predictor.predict(new NDList(subManager.create(state))).get(0);
            int actionData = ActionSampler.sampleMultinomial(prob, random);
            return new DiscreteAction(actionData);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void collect(float[] state, DiscreteAction action, boolean done, float[] nextState, float reward) {
        memory.addTransition(state, action, done, nextState, reward);
    }

    @Override
    public void updateModel() throws TranslateException {
        NDManager subManager = manager.newSubManager();
        MemoryBatch batch = memory.sample(subManager);
        NDArray states = batch.getStates();
        NDArray actions = batch.getActions();

        NDList netOutput = predictor.predict(new NDList(states));

        NDArray distribution = Helper.gather(netOutput.get(0).duplicate(), actions.toIntArray());
        NDArray values = netOutput.get(1).duplicate();

        NDList estimates = estimateAdvantage(values.duplicate(), batch.getRewards());
        NDArray expectedReturns = estimates.get(0);
        NDArray advantages = estimates.get(1);

        int[] index = new int[innerBatchSize];

        for (int i = 0; i < innerUpdates * (1 + batch.size() / innerBatchSize); i++) {
            for (int j = 0; j < innerBatchSize; j++) {
                index[j] = random.nextInt(batch.size());
            }
            NDArray statesSubset = getSample(subManager, states, index);
            NDArray actionsSubset = getSample(subManager, actions, index);
            NDArray distributionSubset = getSample(subManager, distribution, index);
            NDArray expectedReturnsSubset = getSample(subManager, expectedReturns, index);
            NDArray advantagesSubset = getSample(subManager, advantages, index);

            NDList netOutputUpdated = predictor.predict(new NDList(statesSubset));
            NDArray distributionUpdated = Helper.gather(netOutputUpdated.get(0), actionsSubset.toIntArray());
            NDArray valuesUpdated = netOutputUpdated.get(1);

            NDArray lossCritic = (expectedReturnsSubset.sub(valuesUpdated)).square().sum();

            NDArray ratios = distributionUpdated.div(distributionSubset).expandDims(1);

            NDArray lossActor = ratios.clip(ratioLowerBound, ratioUpperBound).mul(advantagesSubset)
                    .minimum(ratios.mul(advantagesSubset)).sum().neg();
            NDArray loss = lossActor.add(lossCritic);

            try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                collector.backward(loss);

                for (Pair<String, Parameter> params : model.getBlock().getParameters()) {
                    NDArray paramsArr = params.getValue().getArray();

                    optimizer.update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                }

            }
        }
    }

    private NDList estimateAdvantage(NDArray values, NDArray rewards) {
        NDArray expected_returns = rewards.duplicate();
        NDArray advantages = rewards.sub(values.squeeze());
        for (long i = expected_returns.getShape().get(0) - 2; i >= 0; i--) {
            NDIndex index = new NDIndex(i);
            expected_returns.set(index, expected_returns.get(i).add(expected_returns.get(i + 1).mul(gamma)));
            advantages.set(index,
                    advantages.get(i).add(values.get(i + 1).add(advantages.get(i + 1).mul(gaeLambda)).mul(gamma)));
        }

        return new NDList(expected_returns, advantages);
    }

    private NDArray getSample(NDManager subManager, NDArray array, int[] index) {
        Shape shape = Shape.update(array.getShape(), 0, innerBatchSize);
        NDArray sample = subManager.zeros(shape, array.getDataType());
        for (int i = 0; i < innerBatchSize; i++) {
            sample.set(new NDIndex(i), array.get(index[i]));
        }
        return sample;
    }
}
