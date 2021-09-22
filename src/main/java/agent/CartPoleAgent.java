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
import env.demo.cartpole.CartPole;
import model.two.CriticValueModel;
import model.two.DiscretePolicyModel;
import utils.ActionSampler;
import utils.Helper;
import utils.MemoryBatch;

import java.util.Arrays;
import java.util.Random;

/**
 * 针对MountainCar的Agent
 *
 * @author Caojunqi
 * @date 2021-09-15 11:09
 */
public class CartPoleAgent extends BaseAgent<DiscreteAction, CartPole> {

    protected Random random = new Random(0);
    protected Optimizer policyOptimizer;
    protected Optimizer valueOptimizer;

    protected NDManager manager = NDManager.newBaseManager();
    protected Model policyModel;
    protected Model valueModel;
    protected Predictor<NDList, NDList> policyPredictor;
    protected Predictor<NDList, NDList> valuePredictor;

    /**
     * GAE参数
     */
    private float gaeLambda;
    private float gamma;
    private float l2Reg = 0.001f;

    /**
     * PPO参数
     */
    private int innerUpdates;
    private int innerBatchSize;
    private float ratioLowerBound;
    private float ratioUpperBound;

    public CartPoleAgent(CartPole env,
                         float gamma, float gaeLambda,
                         float learningRate, int innerUpdates, int innerBatchSize, float ratioClip) {
        super(env);
        this.gamma = gamma;
        this.gaeLambda = gaeLambda;
        this.policyOptimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(learningRate)).build();
        this.valueOptimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(learningRate)).build();
        if (manager != null) {
            manager.close();
        }
        manager = NDManager.newBaseManager();
        policyModel = DiscretePolicyModel.newModel(manager, env.getStateSpaceDim(), env.getActionSpaceDim(), new int[]{3});
        valueModel = CriticValueModel.newModel(manager, env.getStateSpaceDim(), new int[]{3});
        policyPredictor = policyModel.newPredictor(new NoopTranslator());
        valuePredictor = valueModel.newPredictor(new NoopTranslator());
        this.innerUpdates = innerUpdates;
        this.innerBatchSize = innerBatchSize;
        this.ratioLowerBound = 1.0f - ratioClip;
        this.ratioUpperBound = 1.0f + ratioClip;
    }

    @Override
    public DiscreteAction selectAction(float[] state) {
        try (NDManager subManager = manager.newSubManager()) {
            NDArray prob = policyPredictor.predict(new NDList(subManager.create(state))).singletonOrThrow();
            int actionData = ActionSampler.sampleMultinomial(prob, random);
            return new DiscreteAction(actionData);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public DiscreteAction greedyAction(float[] state) {
        try (NDManager subManager = manager.newSubManager()) {
            NDArray prob = policyPredictor.predict(new NDList(subManager.create(state))).singletonOrThrow();
            int actionData = ActionSampler.greedy(prob);
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
    public void updateModel() {
        try (NDManager subManager = manager.newSubManager()) {
            MemoryBatch batch = memory.sample(subManager);
            NDArray states = batch.getStates();
            NDArray actions = batch.getActions();

            NDList policyOutput = policyPredictor.predict(new NDList(states));
            NDArray distribution = Helper.gather(policyOutput.singletonOrThrow().duplicate(), actions.toIntArray());

            NDList valueOutput = valuePredictor.predict(new NDList(states));
            NDArray values = valueOutput.singletonOrThrow().duplicate();

            NDList estimates = estimateAdvantage(values.duplicate(), batch.getRewards(), batch.getMasks());
            NDArray expectedReturns = estimates.get(0);
            NDArray advantages = estimates.get(1);

            int optimIterNum = (int) (states.getShape().get(0) + innerBatchSize - 1) / innerBatchSize;
            for (int i = 0; i < innerUpdates; i++) {
                int[] allIndex = manager.arange((int) states.getShape().get(0)).toIntArray();
                Helper.shuffleArray(allIndex);
                for (int j = 0; j < optimIterNum; j++) {
                    int[] index = Arrays.copyOfRange(allIndex, j * innerBatchSize, Math.min((j + 1) * innerBatchSize, (int) states.getShape().get(0)));
                    NDArray statesSubset = getSample(subManager, states, index);
                    NDArray actionsSubset = getSample(subManager, actions, index);
                    NDArray distributionSubset = getSample(subManager, distribution, index);
                    NDArray expectedReturnsSubset = getSample(subManager, expectedReturns, index);
                    NDArray advantagesSubset = getSample(subManager, advantages, index);

                    // update critic
                    NDList valueOutputUpdated = valuePredictor.predict(new NDList(statesSubset));
                    NDArray valuesUpdated = valueOutputUpdated.singletonOrThrow();
                    NDArray lossCritic = (expectedReturnsSubset.sub(valuesUpdated)).square().mean();
                    for (Pair<String, Parameter> params : valueModel.getBlock().getParameters()) {
                        NDArray paramsArr = params.getValue().getArray();
                        lossCritic = lossCritic.add(paramsArr.square().sum().mul(l2Reg));
                    }
                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(lossCritic);
                        for (Pair<String, Parameter> params : valueModel.getBlock().getParameters()) {
                            NDArray paramsArr = params.getValue().getArray();
                            valueOptimizer.update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                        }
                    }

                    // update policy
                    NDList policyOutputUpdated = policyPredictor.predict(new NDList(statesSubset));
                    NDArray distributionUpdated = Helper.gather(policyOutputUpdated.singletonOrThrow(), actionsSubset.toIntArray());
                    NDArray ratios = distributionUpdated.div(distributionSubset).expandDims(1);

                    NDArray lossActor = ratios.clip(ratioLowerBound, ratioUpperBound).mul(advantagesSubset)
                            .minimum(ratios.mul(advantagesSubset)).mean().neg();
                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(lossActor);
                        for (Pair<String, Parameter> params : policyModel.getBlock().getParameters()) {
                            NDArray paramsArr = params.getValue().getArray();
                            policyOptimizer.update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                        }
                    }
                }
            }
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    private NDList estimateAdvantage(NDArray values, NDArray rewards, NDArray masks) {
        NDManager manager = rewards.getManager();
        NDArray deltas = manager.create(rewards.getShape());
        NDArray advantages = manager.create(rewards.getShape());

        float prevValue = 0;
        float prevAdvantage = 0;
        for (long i = rewards.getShape().get(0) - 1; i >= 0; i--) {
            NDIndex index = new NDIndex(i);
            int mask = masks.getBoolean(i) ? 0 : 1;
            deltas.set(index, rewards.get(i).add(gamma * prevValue * mask).sub(values.get(i)));
            advantages.set(index, deltas.get(i).add(gamma * gaeLambda * prevAdvantage * mask));

            prevValue = values.getFloat(i);
            prevAdvantage = advantages.getFloat(i);
        }

        NDArray expected_returns = values.squeeze().add(advantages);
        NDArray advantagesMean = advantages.mean();
        NDArray advantagesStd = advantages.sub(advantagesMean).pow(2).sum().div(advantages.size() - 1).sqrt();
        advantages = advantages.sub(advantagesMean).div(advantagesStd);

        return new NDList(expected_returns, advantages);
    }

    private NDArray getSample(NDManager subManager, NDArray array, int[] index) {
        Shape shape = Shape.update(array.getShape(), 0, index.length);
        NDArray sample = subManager.zeros(shape, array.getDataType());
        for (int i = 0; i < index.length; i++) {
            sample.set(new NDIndex(i), array.get(index[i]));
        }
        return sample;
    }
}
