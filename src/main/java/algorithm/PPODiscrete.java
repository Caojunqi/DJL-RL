package algorithm;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import env.common.action.impl.DiscreteAction;
import model.model.BasePolicyModel;
import model.model.BaseValueModel;
import model.model.CriticValueModel;
import model.model.DiscretePolicyModel;
import resource.ConstantParameter;
import utils.ActionSampler;
import utils.Helper;
import utils.MemoryBatch;

import java.util.Arrays;

/**
 * PPO算法  Proximal policy optimization algorithms
 * 针对离散型动作空间
 *
 * @author Caojunqi
 * @date 2021-09-16 15:44
 */
public class PPODiscrete extends BaseAlgorithm<DiscreteAction> {
    /**
     * 策略模型
     */
    protected BasePolicyModel<DiscreteAction> policyModel;
    /**
     * 价值函数近似模型
     */
    protected BaseValueModel valueModel;

    public PPODiscrete(int stateDim, int actionDim) {
        this.policyModel = DiscretePolicyModel.newModel(manager, stateDim, actionDim);
        this.valueModel = CriticValueModel.newModel(manager, stateDim);
    }

    @Override
    public DiscreteAction selectAction(float[] state) {
        // 此处将单一状态数组转为多维的，这样可以保证在predict过程中，传入1个状态和传入多个状态，输入数据的维度是一致的。
        float[][] states = new float[][]{state};
        try (NDManager subManager = manager.newSubManager()) {
            NDArray prob = policyModel.getPredictor().predict(new NDList(subManager.create(states))).singletonOrThrow();
            int actionData = ActionSampler.sampleMultinomial(prob, random);
            return new DiscreteAction(actionData);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public DiscreteAction greedyAction(float[] state) {
        // 此处将单一状态数组转为多维的，这样可以保证在predict过程中，传入1个状态和传入多个状态，输入数据的维度是一致的。
        float[][] states = new float[][]{state};
        try (NDManager subManager = manager.newSubManager()) {
            NDArray prob = policyModel.getPredictor().predict(new NDList(subManager.create(states))).singletonOrThrow();
            int actionData = ActionSampler.greedy(prob);
            return new DiscreteAction(actionData);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void updateModel() {
        try (NDManager subManager = manager.newSubManager()) {
            MemoryBatch batch = memory.sample(subManager);
            NDArray states = batch.getStates();
            NDArray actions = batch.getActions();

            NDList policyOutput = policyModel.getPredictor().predict(new NDList(states));
            NDArray distribution = Helper.gather(policyOutput.singletonOrThrow().duplicate(), actions.toIntArray());

            NDList valueOutput = valueModel.getPredictor().predict(new NDList(states));
            NDArray values = valueOutput.singletonOrThrow().duplicate();

            NDList estimates = estimateAdvantage(values.duplicate(), batch.getRewards(), batch.getMasks());
            NDArray expectedReturns = estimates.get(0);
            NDArray advantages = estimates.get(1);

            int optimIterNum = (int) (states.getShape().get(0) + ConstantParameter.INNER_BATCH_SIZE - 1) / ConstantParameter.INNER_BATCH_SIZE;
            for (int i = 0; i < ConstantParameter.INNER_UPDATES; i++) {
                int[] allIndex = manager.arange((int) states.getShape().get(0)).toIntArray();
                Helper.shuffleArray(allIndex);
                for (int j = 0; j < optimIterNum; j++) {
                    int[] index = Arrays.copyOfRange(allIndex, j * ConstantParameter.INNER_BATCH_SIZE, Math.min((j + 1) * ConstantParameter.INNER_BATCH_SIZE, (int) states.getShape().get(0)));
                    NDArray statesSubset = getSample(subManager, states, index);
                    NDArray actionsSubset = getSample(subManager, actions, index);
                    NDArray distributionSubset = getSample(subManager, distribution, index);
                    NDArray expectedReturnsSubset = getSample(subManager, expectedReturns, index);
                    NDArray advantagesSubset = getSample(subManager, advantages, index);

                    // update critic
                    NDList valueOutputUpdated = valueModel.getPredictor().predict(new NDList(statesSubset));
                    NDArray valuesUpdated = valueOutputUpdated.singletonOrThrow();
                    NDArray lossCritic = (expectedReturnsSubset.sub(valuesUpdated)).square().mean();
                    for (Pair<String, Parameter> params : valueModel.getModel().getBlock().getParameters()) {
                        NDArray paramsArr = params.getValue().getArray();
                        lossCritic = lossCritic.add(paramsArr.square().sum().mul(ConstantParameter.L2_REG));
                    }
                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(lossCritic);
                        for (Pair<String, Parameter> params : valueModel.getModel().getBlock().getParameters()) {
                            NDArray paramsArr = params.getValue().getArray();
                            valueModel.getOptimizer().update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                        }
                    }

                    // update policy
                    NDList policyOutputUpdated = policyModel.getPredictor().predict(new NDList(statesSubset));
                    NDArray distributionUpdated = Helper.gather(policyOutputUpdated.singletonOrThrow(), actionsSubset.toIntArray());
                    NDArray ratios = distributionUpdated.div(distributionSubset).expandDims(1);

                    NDArray surr1 = ratios.mul(advantagesSubset);
                    NDArray surr2 = ratios.clip(ConstantParameter.RATIO_LOWER_BOUND, ConstantParameter.RATIO_UPPER_BOUND).mul(advantagesSubset);
                    NDArray lossActor = surr1.minimum(surr2).mean().neg();

                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(lossActor);
                        for (Pair<String, Parameter> params : policyModel.getModel().getBlock().getParameters()) {
                            NDArray paramsArr = params.getValue().getArray();
                            policyModel.getOptimizer().update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                        }
                    }
                }
            }
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }
}
