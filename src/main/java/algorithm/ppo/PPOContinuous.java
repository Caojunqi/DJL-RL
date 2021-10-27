package algorithm.ppo;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import algorithm.BaseAlgorithm;
import algorithm.CommonParameter;
import algorithm.ppo.model.BasePolicyModel;
import algorithm.ppo.model.BaseValueModel;
import algorithm.ppo.model.BoxPolicyModel;
import algorithm.ppo.model.CriticValueModel;
import env.common.action.impl.BoxAction;
import utils.Helper;
import utils.MemoryBatch;

import java.util.Arrays;

/**
 * PPO算法  Proximal policy optimization algorithms
 * 针对连续型动作空间
 *
 * @author Caojunqi
 * @date 2021-10-08 21:53
 */
public class PPOContinuous extends BaseAlgorithm<BoxAction> {
    /**
     * 策略模型
     */
    protected BasePolicyModel<BoxAction> policyModel;
    /**
     * 价值函数近似模型
     */
    protected BaseValueModel valueModel;

    public PPOContinuous(int stateDim, int actionDim) {
        this.policyModel = BoxPolicyModel.newModel(manager, stateDim, actionDim);
        this.valueModel = CriticValueModel.newModel(manager, stateDim);
    }

    @Override
    public BoxAction selectAction(float[] state) {
        // 此处将单一状态数组转为多维的，这样可以保证在predict过程中，传入1个状态和传入多个状态，输入数据的维度是一致的。
        float[][] states = new float[][]{state};
        NDManager subManager = manager.newSubManager();
        return policyModel.policy(new NDList(subManager.create(states)), false, false).singletonOrThrow();
    }

    @Override
    public BoxAction greedyAction(float[] state) {
        // 此处将单一状态数组转为多维的，这样可以保证在predict过程中，传入1个状态和传入多个状态，输入数据的维度是一致的。
        float[][] states = new float[][]{state};
        NDManager subManager = manager.newSubManager();
        return policyModel.policy(new NDList(subManager.create(states)), true, false).singletonOrThrow();
    }

    @Override
    public void updateModel() {
        try (NDManager subManager = manager.newSubManager()) {
            MemoryBatch batch = memory.sample(subManager);
            NDArray states = batch.getStates();
            NDArray actions = batch.getActions();

            NDList policyOutput = policyModel.getPredictor().predict(new NDList(states));
            NDArray distribution = normalLogDensity(actions, policyOutput.get(0).duplicate(), policyOutput.get(1).duplicate(), policyOutput.get(2).duplicate());
            distribution = distribution.exp();

            NDList valueOutput = valueModel.getPredictor().predict(new NDList(states));
            NDArray values = valueOutput.singletonOrThrow().duplicate();

            NDList estimates = estimateAdvantage(values.duplicate(), batch.getRewards(), batch.getMasks());
            NDArray expectedReturns = estimates.get(0);
            NDArray advantages = estimates.get(1);

            int optimIterNum = (int) (states.getShape().get(0) + CommonParameter.INNER_BATCH_SIZE - 1) / CommonParameter.INNER_BATCH_SIZE;
            for (int i = 0; i < CommonParameter.INNER_UPDATES; i++) {
                int[] allIndex = manager.arange((int) states.getShape().get(0)).toIntArray();
                Helper.shuffleArray(allIndex);
                for (int j = 0; j < optimIterNum; j++) {
                    int[] index = Arrays.copyOfRange(allIndex, j * CommonParameter.INNER_BATCH_SIZE, Math.min((j + 1) * CommonParameter.INNER_BATCH_SIZE, (int) states.getShape().get(0)));
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
                        lossCritic = lossCritic.add(paramsArr.square().sum().mul(CommonParameter.L2_REG));
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
                    NDArray distributionUpdated = normalLogDensity(actionsSubset, policyOutputUpdated.get(0), policyOutputUpdated.get(1), policyOutputUpdated.get(2));
                    distributionUpdated = distributionUpdated.exp();
                    NDArray ratios = distributionUpdated.div(distributionSubset);

                    NDArray surr1 = ratios.mul(advantagesSubset);
                    NDArray surr2 = ratios.clip(PPOParameter.RATIO_LOWER_BOUND, PPOParameter.RATIO_UPPER_BOUND).mul(advantagesSubset);
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

    private NDArray normalLogDensity(NDArray actions, NDArray actionMean, NDArray actionLogStd, NDArray actionStd) {
        NDArray var = actionStd.pow(2);
        NDArray logDensity = actions.sub(actionMean).pow(2).div(var.mul(2)).neg().sub(Math.log(2 * Math.PI) * 0.5).sub(actionLogStd);
        return logDensity.sum(new int[]{1}, true);
    }
}
