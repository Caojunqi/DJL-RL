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
import algorithm.sac.model.GaussianPolicyModel;
import algorithm.sac.model.QFunctionModel;
import env.common.action.impl.BoxAction;
import utils.Helper;
import utils.MemoryBatch;
import utils.datatype.PolicyPair;

import java.util.Arrays;

/**
 * SAC算法  Soft Actor-Critic algorithm
 * 针对连续型动作空间和连续型状态空间
 *
 * @author Caojunqi
 * @date 2021-10-12 15:03
 */
public class SACContinuous extends BaseAlgorithm<BoxAction> {
    /**
     * 策略模型
     */
    private BasePolicyModel<BoxAction> policyModel;
    private Optimizer policyOptimizer;
    /**
     * Q函数模型_1
     */
    private BaseModel qf1;
    private Optimizer qfOptimizer1;
    /**
     * Q函数模型_2
     */
    private BaseModel qf2;
    private Optimizer qfOptimizer2;
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

    public SACContinuous(int stateDim, int actionDim) {
        this.policyModel = GaussianPolicyModel.newModel(manager, stateDim, actionDim);
        this.policyOptimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(SACParameter.POLICY_LR)).optWeightDecays(SACParameter.POLICY_WEIGHT_DECAY).build();
        this.qf1 = QFunctionModel.newModel(manager, stateDim, actionDim);
        this.qfOptimizer1 = Optimizer.adam().optLearningRateTracker(Tracker.fixed(SACParameter.QF_LR)).build();
        this.qf2 = QFunctionModel.newModel(manager, stateDim, actionDim);
        this.qfOptimizer2 = Optimizer.adam().optLearningRateTracker(Tracker.fixed(SACParameter.QF_LR)).build();
        this.targetQf1 = QFunctionModel.newModel(manager, stateDim, actionDim);
        this.targetQf2 = QFunctionModel.newModel(manager, stateDim, actionDim);

        Helper.copyModel(qf1, targetQf1);
        Helper.copyModel(qf2, targetQf2);
        this.entropyScale = manager.create(SACParameter.ENTROPY_SCALE);
        this.tgtEntro = manager.create(-actionDim);
        this.logAlpha = manager.zeros(new Shape(1));
        this.logAlpha.setRequiresGradient(true);
        this.alphasOptimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(SACParameter.POLICY_LR)).build();
    }

    @Override
    public BoxAction selectAction(float[] state) {
        // 此处将单一状态数组转为多维的，这样可以保证在predict过程中，传入1个状态和传入多个状态，输入数据的维度是一致的。
        float[][] states = new float[][]{state};
        NDManager subManager = manager.newSubManager();
        return policyModel.policy(new NDList(subManager.create(states)), false, false, true).singletonOrThrow();
    }

    @Override
    public BoxAction greedyAction(float[] state) {
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

            float policyPriorLogProb = 0.0f; // Uniform prior // TODO: Normal prior

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
                    PolicyPair<BoxAction> policyPair = this.policyModel.policy(new NDList(statesSubset), false, true, false);
                    NDArray newActions = policyPair.getInfo().get(0);
                    NDArray newLogPi = policyPair.getInfo().get(4);
                    PolicyPair<BoxAction> nextPolicyPair = this.policyModel.policy(new NDList(nextStatesSubset), false, true, true);
                    NDArray nextActions = nextPolicyPair.getInfo().get(0);
                    NDArray nextLogPi = nextPolicyPair.getInfo().get(4);

                    // =========== Policy Evaluation Step ============

                    // Estimate from target Q-value(s)
                    // Q1_target(s', a')
                    NDArray nextStatesActions = nextStatesSubset.concat(nextActions, -1).toType(DataType.FLOAT32, false);
                    NDArray nextQ1 = this.targetQf1.getPredictor().predict(new NDList(nextStatesActions)).singletonOrThrow().duplicate();
                    // Q2_target(s', a')
                    NDArray nextQ2 = this.targetQf2.getPredictor().predict(new NDList(nextStatesActions)).singletonOrThrow().duplicate();
                    // Minimum Unintentional Double-Q
                    NDArray nextQ = nextQ1.minimum(nextQ2);
                    // V_target(s')
                    NDArray nextV = nextQ.sub(alpha.mul(nextLogPi)).duplicate();

                    // Calculate Bellman Backup for Q-values
                    NDArray qBackup = rewardsSubset.add(terminationsSubset.neg().add(1).mul(CommonParameter.GAMMA).mul(nextV));

                    // Prediction Q(s,a)
                    NDArray statesActions = statesSubset.concat(actionsSubset, -1).toType(DataType.FLOAT32, false);
                    NDArray predQ1 = this.qf1.getPredictor().predict(new NDList(statesActions)).singletonOrThrow();
                    // Critic loss: Mean Squared Bellman Error (MSBE)
                    NDArray lossQf1 = predQ1.sub(qBackup).pow(2).mean().mul(0.5).squeeze(-1);

                    NDArray predQ2 = this.qf2.getPredictor().predict(new NDList(statesActions)).singletonOrThrow();
                    NDArray lossQf2 = predQ2.sub(qBackup).pow(2).mean().mul(0.5).squeeze(-1);

                    NDArray qvaluesLoss = lossQf1.add(lossQf2);
                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(qvaluesLoss);
                        for (Pair<String, Parameter> params : qf1.getModel().getBlock().getParameters()) {
                            NDArray paramsArr = params.getValue().getArray();
                            qfOptimizer1.update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                        }
                        for (Pair<String, Parameter> params : qf2.getModel().getBlock().getParameters()) {
                            NDArray paramsArr = params.getValue().getArray();
                            qfOptimizer2.update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                        }
                    }

                    // =========== Policy Improvement Step ============

                    // TODO: Decide if use the minimum btw q1 and q2. Using new_q1 for now
                    NDArray statesNewActions = statesSubset.concat(newActions, -1).toType(DataType.FLOAT32, false);
                    NDArray newQ1 = this.qf1.getPredictor().predict(new NDList(statesNewActions)).singletonOrThrow();
                    NDArray newQ = newQ1;

                    // Policy KL loss: - (E_a[Q(s, a) + H(.)])
                    NDArray policyKlLoss = newQ.sub(alpha.mul(newLogPi)).add(policyPriorLogProb).mean().neg();
                    // TODO: It can include regularization of mean, std
                    double policyReguLoss = 0;
                    NDArray policyLoss = policyKlLoss.add(policyReguLoss).sum();
                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(policyLoss);
                        for (Pair<String, Parameter> params : policyModel.getModel().getBlock().getParameters()) {
                            NDArray paramsArr = params.getValue().getArray();
                            policyOptimizer.update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                        }
                    }

                    // =========== Entropy Adjustment Step ===========

                    // NOTE: In formula is alphas and not log_alphas
                    NDArray alphasLoss = this.logAlpha.mul(newLogPi.squeeze(-1).add(this.tgtEntro).mean().duplicate()).neg();
                    NDArray hiuAlphasLoss = alphasLoss.sum();
                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(hiuAlphasLoss);
                        alphasOptimizer.update(this.logAlpha.getUid(), this.logAlpha, this.logAlpha.getGradient().duplicate());
                    }
                    this.logAlpha.clip(Math.log(SACParameter.MIN_ALPHA), Math.log(SACParameter.MAX_ALPHA));

                    // =========== Target Networks Update Step ===========

                    Helper.softParamUpdateFromTo(this.qf1, this.targetQf1, CommonParameter.SOFT_TARGET_TAU);
                    Helper.softParamUpdateFromTo(this.qf2, this.targetQf2, CommonParameter.SOFT_TARGET_TAU);
                }
            }
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }
}
