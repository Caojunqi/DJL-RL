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
import model.CriticValueModel;
import model.DiscretePolicyModel;
import utils.ActionSampler;
import utils.Helper;
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
    protected Optimizer optimizer;

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

    /**
     * PPO参数
     */
    private int innerUpdates;
    private int innerBatchSize;
    private float ratioLowerBound;
    private float ratioUpperBound;

    public MountainCarAgent(MountainCar env,
                            float gamma, float gaeLambda,
                            float learningRate, int innerUpdates, int innerBatchSize, float ratioClip) {
        super(env);
        this.gamma = gamma;
        this.gaeLambda = gaeLambda;
        this.optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(learningRate)).build();
        if (manager != null) {
            manager.close();
        }
        manager = NDManager.newBaseManager();
        policyModel = DiscretePolicyModel.newModel(manager, env.getStateSpaceDim(), env.getActionSpaceDim());
        valueModel = CriticValueModel.newModel(manager, env.getStateSpaceDim());
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

                NDList policyOutputUpdated = policyPredictor.predict(new NDList(statesSubset));
                NDList valueOutputUpdated = valuePredictor.predict(new NDList(statesSubset));
                NDArray distributionUpdated = Helper.gather(policyOutputUpdated.singletonOrThrow(), actionsSubset.toIntArray());
                NDArray valuesUpdated = valueOutputUpdated.singletonOrThrow();

                NDArray lossCritic = (expectedReturnsSubset.sub(valuesUpdated)).square().sum();

                NDArray ratios = distributionUpdated.div(distributionSubset).expandDims(1);

                NDArray lossActor = ratios.clip(ratioLowerBound, ratioUpperBound).mul(advantagesSubset)
                        .minimum(ratios.mul(advantagesSubset)).sum().neg();
//                NDArray loss = lossActor.add(lossCritic);

                // update critic
                try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                    collector.backward(lossCritic);
                    for (Pair<String, Parameter> params : valueModel.getBlock().getParameters()) {
                        NDArray paramsArr = params.getValue().getArray();
                        optimizer.update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                    }
                }

                // update policy
                try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                    collector.backward(lossActor);
                    for (Pair<String, Parameter> params : policyModel.getBlock().getParameters()) {
                        NDArray paramsArr = params.getValue().getArray();
                        optimizer.update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                    }
                }
            }
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
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
