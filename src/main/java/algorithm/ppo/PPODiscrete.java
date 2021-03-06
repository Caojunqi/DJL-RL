package algorithm.ppo;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import algorithm.BaseAlgorithm;
import algorithm.CommonParameter;
import algorithm.ppo.model.BasePolicyModel;
import algorithm.ppo.model.BaseValueModel;
import env.action.core.impl.DiscreteAction;
import env.state.core.IState;
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
public class PPODiscrete<S extends IState<S>> extends BaseAlgorithm<S, DiscreteAction> {
    /**
     * 策略模型
     */
    private BasePolicyModel<DiscreteAction> policyModel;
    private Optimizer policyOptimizer;
    /**
     * 价值函数近似模型
     */
    private BaseValueModel valueModel;
    private Optimizer valueOptimizer;

    public PPODiscrete(NDManager manager, BasePolicyModel<DiscreteAction> policyModel, BaseValueModel valueModel) {
        super(manager);
        this.policyModel = policyModel;
        this.policyOptimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(CommonParameter.LEARNING_RATE)).build();
        this.valueModel = valueModel;
        this.valueOptimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(CommonParameter.LEARNING_RATE)).build();
    }

    @Override
    public DiscreteAction selectAction(S state) {
        NDManager subManager = manager.newSubManager();
        return policyModel.policy(state.singleStateList(subManager), false, false, true).singletonOrThrow();
    }

    @Override
    public DiscreteAction greedyAction(S state) {
        NDManager subManager = manager.newSubManager();
        return policyModel.policy(state.singleStateList(subManager), true, false, true).singletonOrThrow();
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
                            valueOptimizer.update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                        }
                    }

                    // update policy
                    NDList policyOutputUpdated = policyModel.getPredictor().predict(new NDList(statesSubset));
                    NDArray distributionUpdated = Helper.gather(policyOutputUpdated.singletonOrThrow(), actionsSubset.toIntArray());
                    NDArray ratios = distributionUpdated.div(distributionSubset);

                    NDArray surr1 = ratios.mul(advantagesSubset);
                    NDArray surr2 = ratios.clip(PPOParameter.RATIO_LOWER_BOUND, PPOParameter.RATIO_UPPER_BOUND).mul(advantagesSubset);
                    NDArray lossActor = surr1.minimum(surr2).mean().neg();

                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(lossActor);
                        for (Pair<String, Parameter> params : policyModel.getModel().getBlock().getParameters()) {
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
}
