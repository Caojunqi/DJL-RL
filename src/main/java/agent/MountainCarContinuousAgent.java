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
import env.common.action.impl.BoxAction;
import env.demo.mountaincar.MountainCarContinuous;
import model.block.BoxPolicyModelBlock;
import model.block.CriticValueModelBlock;
import utils.ActionSampler;
import utils.Helper;
import utils.MemoryBatch;

import java.util.Arrays;
import java.util.Random;

/**
 * 针对MountainCarContinuous的Agent
 *
 * @author Caojunqi
 * @date 2021-09-23 11:57
 */
public class MountainCarContinuousAgent extends BaseAgent<BoxAction, MountainCarContinuous> {

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

    public MountainCarContinuousAgent(MountainCarContinuous env,
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
        policyModel = BoxPolicyModelBlock.newModel(manager, env.getStateSpaceDim(), env.getActionSpaceDim());
        valueModel = CriticValueModelBlock.newModel(manager, env.getStateSpaceDim());
        policyPredictor = policyModel.newPredictor(new NoopTranslator());
        valuePredictor = valueModel.newPredictor(new NoopTranslator());
        this.innerUpdates = innerUpdates;
        this.innerBatchSize = innerBatchSize;
        this.ratioLowerBound = 1.0f - ratioClip;
        this.ratioUpperBound = 1.0f + ratioClip;
    }

    @Override
    public BoxAction selectAction(float[] state) {
        // 此处将单一状态数组转为多维的，这样可以保证在predict过程中，传入1个状态和传入多个状态，输入数据的维度是一致的。
        float[][] states = new float[][]{state};
        try (NDManager subManager = manager.newSubManager()) {
            NDList distribution = policyPredictor.predict(new NDList(subManager.create(states)));
            double[] actionData = ActionSampler.sampleNormal(distribution.get(0), distribution.get(2), random);
            return new BoxAction(actionData);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public BoxAction greedyAction(float[] state) {
        // todo 待完善
        return null;
    }

    public NDArray normalLogDensity(NDArray actions, NDArray actionMean, NDArray actionLogStd, NDArray actionStd) {
        NDArray var = actionStd.pow(2);
        NDArray logDensity = actions.sub(actionMean).pow(2).div(var.mul(2)).neg().sub(Math.log(2 * Math.PI) * 0.5).sub(actionLogStd);
        return logDensity.sum(new int[]{1}, true);
    }

    @Override
    public void collect(float[] state, BoxAction action, boolean done, float[] nextState, float reward) {
        memory.addTransition(state, action, done, nextState, reward);
    }

    @Override
    public void updateModel() {
        try (NDManager subManager = manager.newSubManager()) {
            MemoryBatch batch = memory.sample(subManager);
            NDArray states = batch.getStates();
            NDArray actions = batch.getActions();

            NDList policyOutput = policyPredictor.predict(new NDList(states));
            NDArray distribution = normalLogDensity(actions, policyOutput.get(0).duplicate(), policyOutput.get(1).duplicate(), policyOutput.get(2).duplicate());
            distribution = distribution.exp();

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
                    NDArray distributionUpdated = normalLogDensity(actionsSubset, policyOutputUpdated.get(0), policyOutputUpdated.get(1), policyOutputUpdated.get(2));
                    distributionUpdated = distributionUpdated.exp();
                    NDArray ratios = distributionUpdated.div(distributionSubset);

                    NDArray surr1 = ratios.mul(advantagesSubset);
                    NDArray surr2 = ratios.clip(ratioLowerBound, ratioUpperBound).mul(advantagesSubset);
                    NDArray lossActor = surr1.minimum(surr2).mean().neg();

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
        NDArray deltas = manager.create(rewards.getShape().add(1));
        NDArray advantages = manager.create(rewards.getShape().add(1));

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

        NDArray expected_returns = values.add(advantages);
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
