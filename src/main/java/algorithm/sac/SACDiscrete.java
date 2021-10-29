package algorithm.sac;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import algorithm.BaseAlgorithm;
import algorithm.BaseModel;
import algorithm.CommonParameter;
import algorithm.ppo.model.BasePolicyModel;
import algorithm.sac.model.DiscreteGaussianPolicyModel;
import algorithm.sac.model.DiscreteQFunctionModel;
import env.common.action.impl.DiscreteAction;
import utils.Helper;
import utils.MemoryBatch;
import utils.datatype.PolicyPair;

import java.util.Arrays;

/**
 * SAC算法  Soft Actor-Critic algorithm
 * 针对离散型动作空间
 *
 * @author Caojunqi
 * @date 2021-10-26 11:26
 */
public class SACDiscrete extends BaseAlgorithm<DiscreteAction> {

    /**
     * 策略模型
     */
    private BasePolicyModel<DiscreteAction> policyModel;
    /**
     * Q函数模型_1
     */
    private BaseModel qf1;
    /**
     * Q函数模型_2
     */
    private BaseModel qf2;
    /**
     * 目标Q函数模型_1
     */
    private BaseModel targetQf1;
    /**
     * 目标Q函数模型_2
     */
    private BaseModel targetQf2;


    private NDArray entropyScale;
    private NDArray tgtEntro;
    private NDArray logAlpha;
    private Optimizer alphasOptimizer;

    public SACDiscrete(int stateDim, int actionDim) {
        this.policyModel = DiscreteGaussianPolicyModel.newModel(manager, stateDim, actionDim);
        this.qf1 = DiscreteQFunctionModel.newModel(manager, stateDim, actionDim);
        this.qf2 = DiscreteQFunctionModel.newModel(manager, stateDim, actionDim);
        this.targetQf1 = DiscreteQFunctionModel.newModel(manager, stateDim, actionDim);
        this.targetQf2 = DiscreteQFunctionModel.newModel(manager, stateDim, actionDim);

        Helper.copyModel(qf1, targetQf1);
        Helper.copyModel(qf2, targetQf2);
        this.entropyScale = manager.create(SACParameter.ENTROPY_SCALE);
        this.tgtEntro = manager.create(-actionDim);
        this.logAlpha = manager.zeros(new Shape(1));
        this.logAlpha.setRequiresGradient(true);
        this.alphasOptimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(SACParameter.POLICY_LR)).build();
    }

    @Override
    public DiscreteAction selectAction(float[] state) {
        // 此处将单一状态数组转为多维的，这样可以保证在predict过程中，传入1个状态和传入多个状态，输入数据的维度是一致的。
        float[][] states = new float[][]{state};
        NDManager subManager = manager.newSubManager();
        return policyModel.policy(new NDList(subManager.create(states)), false, false, true).singletonOrThrow();
    }

    @Override
    public DiscreteAction greedyAction(float[] state) {
        // 此处将单一状态数组转为多维的，这样可以保证在predict过程中，传入1个状态和传入多个状态，输入数据的维度是一致的。
        float[][] states = new float[][]{state};
        NDManager subManager = manager.newSubManager();
        return policyModel.policy(new NDList(subManager.create(states)), true, false, true).singletonOrThrow();
    }

    @Override
    public void updateModel() {
        try (NDManager subManager = manager.newSubManager()) {
            MemoryBatch batch = memory.sample(subManager);
            NDArray states = batch.getStates();
            NDArray actions = batch.getActions();
            NDArray nextStates = batch.getNextStates();
            NDArray rewards = batch.getRewards();
            NDArray masks = batch.getMasks();
            NDArray terminations = masks.toType(DataType.FLOAT64, true);

            int optimIterNum = (int) (states.getShape().get(0) + CommonParameter.INNER_BATCH_SIZE - 1) / CommonParameter.INNER_BATCH_SIZE;
            for (int i = 0; i < CommonParameter.INNER_UPDATES; i++) {
                int[] allIndex = manager.arange((int) states.getShape().get(0)).toIntArray();
                Helper.shuffleArray(allIndex);
                for (int j = 0; j < optimIterNum; j++) {
                    int[] index = Arrays.copyOfRange(allIndex, j * CommonParameter.INNER_BATCH_SIZE, Math.min((j + 1) * CommonParameter.INNER_BATCH_SIZE, (int) states.getShape().get(0)));
                    NDArray statesSubset = getSample(subManager, states, index);
                    NDArray actionsSubset = getSample(subManager, actions, index);
                    NDArray nextStatesSubset = getSample(subManager, nextStates, index);
                    NDArray rewardsSubset = getSample(subManager, rewards, index);
                    NDArray terminationsSubset = getSample(subManager, terminations, index);


                    // Alphas
                    NDArray alpha = this.entropyScale.mul(this.logAlpha.exp().duplicate());

                    // Actions for batch observation
                    PolicyPair<DiscreteAction> nextPolicyPair = this.policyModel.policy(new NDList(nextStatesSubset), false, true, false);
                    NDArray nextDistribution = nextPolicyPair.getInfo().get(0).duplicate();
                    NDArray nextLogDistribution = nextPolicyPair.getInfo().get(1).duplicate();

                    // =========== Policy Evaluation Step ============

                    NDArray nextTargetQ1 = this.targetQf1.getPredictor().predict(new NDList(nextStatesSubset)).singletonOrThrow().duplicate();
                    NDArray nextTargetQ2 = this.targetQf2.getPredictor().predict(new NDList(nextStatesSubset)).singletonOrThrow().duplicate();
                    NDArray nextTargetMinQf = nextDistribution.mul(nextTargetQ1.minimum(nextTargetQ2).sub(alpha.mul(nextLogDistribution))).duplicate();
                    nextTargetMinQf = nextTargetMinQf.sum(new int[]{-1}, true);
                    NDArray nextQValue = rewardsSubset.add(terminationsSubset.neg().add(1).mul(CommonParameter.GAMMA).mul(nextTargetMinQf));

                    // Prediction Q(s,a)
                    NDArray predQ1 = this.qf1.getPredictor().predict(new NDList(statesSubset)).singletonOrThrow();
                    predQ1 = Helper.gather(predQ1, actionsSubset.toIntArray());
                    // Critic loss: Mean Squared Bellman Error (MSBE)
                    NDArray lossQf1 = predQ1.sub(nextQValue).pow(2).mean();
                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(lossQf1);
                        for (Pair<String, Parameter> params : qf1.getModel().getBlock().getParameters()) {
                            NDArray paramsArr = params.getValue().getArray();
                            qf1.getOptimizer().update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                        }
                    }

                    NDArray predQ2 = this.qf2.getPredictor().predict(new NDList(statesSubset)).singletonOrThrow();
                    predQ2 = Helper.gather(predQ2, actionsSubset.toIntArray());
                    NDArray lossQf2 = predQ2.sub(nextQValue).pow(2).mean();
                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(lossQf2);
                        for (Pair<String, Parameter> params : qf2.getModel().getBlock().getParameters()) {
                            NDArray paramsArr = params.getValue().getArray();
                            qf2.getOptimizer().update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                        }
                    }

                    // =========== Target Networks Update Step ===========

                    Helper.softParamUpdateFromTo(this.qf1, this.targetQf1, CommonParameter.SOFT_TARGET_TAU);
                    Helper.softParamUpdateFromTo(this.qf2, this.targetQf2, CommonParameter.SOFT_TARGET_TAU);

                    // =========== Policy Improvement Step ============

                    PolicyPair<DiscreteAction> newPolicyPair = this.policyModel.policy(new NDList(statesSubset), false, true, false);
                    NDArray newDistribution = newPolicyPair.getInfo().get(0);
                    NDArray newLogDistribution = newPolicyPair.getInfo().get(1);

                    NDArray newQ1 = this.qf1.getPredictor().predict(new NDList(statesSubset)).singletonOrThrow();
                    NDArray newQ2 = this.qf2.getPredictor().predict(new NDList(statesSubset)).singletonOrThrow();
                    NDArray minNewQ = newQ1.minimum(newQ2);

                    NDArray insideTerm = alpha.mul(newLogDistribution).sub(minNewQ);
                    NDArray policyLoss = insideTerm.mul(newDistribution).sum(new int[]{1}).mean();
                    newLogDistribution = newLogDistribution.mul(newDistribution).sum(new int[]{1});
                    NDArray alphaLoss = this.logAlpha.mul(newLogDistribution.add(this.tgtEntro).duplicate()).mean();

                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(policyLoss);
                        for (Pair<String, Parameter> params : policyModel.getModel().getBlock().getParameters()) {
                            NDArray paramsArr = params.getValue().getArray();
                            policyModel.getOptimizer().update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                        }
                    }

                    // =========== Entropy Adjustment Step ===========

                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(alphaLoss);
                        alphasOptimizer.update(this.logAlpha.getUid(), this.logAlpha, this.logAlpha.getGradient().duplicate());
                    }
                    this.logAlpha.clip(Math.log(SACParameter.MIN_ALPHA), Math.log(SACParameter.MAX_ALPHA));
                }
            }
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }
}
