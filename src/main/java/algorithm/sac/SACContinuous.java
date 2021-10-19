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

    public SACContinuous(int stateDim, int actionDim) {
        this.policyModel = GaussianPolicyModel.newModel(manager, stateDim, actionDim);
        this.qf1 = QFunctionModel.newModel(manager, stateDim, actionDim);
        this.qf2 = QFunctionModel.newModel(manager, stateDim, actionDim);
        this.targetQf1 = QFunctionModel.newModel(manager, stateDim, actionDim);
        this.targetQf2 = QFunctionModel.newModel(manager, stateDim, actionDim);

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
        return policyModel.policy(new NDList(subManager.create(states)), false, false).getAction();
    }

    @Override
    public BoxAction greedyAction(float[] state) {
        // 此处将单一状态数组转为多维的，这样可以保证在predict过程中，传入1个状态和传入多个状态，输入数据的维度是一致的。
        float[][] states = new float[][]{state};
        NDManager subManager = manager.newSubManager();
        return policyModel.policy(new NDList(subManager.create(states)), true, false).getAction();
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

            float policyPriorLogProb = 0.0f; // Uniform prior // TODO: Normal prior

            // Alphas
            NDArray alpha = this.entropyScale.mul(this.logAlpha.exp());

            // Actions for batch observation
            PolicyPair<BoxAction> policyPair = this.policyModel.policy(new NDList(states), false, true);
            NDArray newActions = subManager.create(policyPair.getAction().getActionData()).expandDims(-1);
            NDArray newLogPi = policyPair.getInfo().singletonOrThrow();
            PolicyPair<BoxAction> nextPolicyPair = this.policyModel.policy(new NDList(nextStates), false, true);
            NDArray nextActions = subManager.create(nextPolicyPair.getAction().getActionData()).expandDims(-1);
            NDArray nextLogPi = nextPolicyPair.getInfo().singletonOrThrow();

            // =========== Policy Evaluation Step ============

            // Estimate from target Q-value(s)
            // Q1_target(s', a')
            NDArray nextStatesActions = nextStates.concat(nextActions, -1).toType(DataType.FLOAT32, false);
            NDArray nextQ1 = this.targetQf1.getPredictor().predict(new NDList(nextStatesActions)).singletonOrThrow();
            // Q2_target(s', a')
            NDArray nextQ2 = this.targetQf1.getPredictor().predict(new NDList(nextStatesActions)).singletonOrThrow();
            // Minimum Unintentional Double-Q
            NDArray nextQ = nextQ1.minimum(nextQ2);
            // V_target(s')
            NDArray nextV = nextQ.sub(alpha.mul(nextLogPi));

            // Calculate Bellman Backup for Q-values
            NDArray terminations = masks.toType(DataType.FLOAT64, true);
            NDArray qBackup = rewards.add(terminations.neg().add(1).mul(CommonParameter.GAMMA).mul(nextV));

            // Prediction Q(s,a)
            NDArray statesActions = states.concat(actions, -1).toType(DataType.FLOAT32, false);
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
                    qf1.getOptimizer().update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                }
                for (Pair<String, Parameter> params : qf2.getModel().getBlock().getParameters()) {
                    NDArray paramsArr = params.getValue().getArray();
                    qf2.getOptimizer().update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                }
            }

            // =========== Policy Improvement Step ============

            // TODO: Decide if use the minimum btw q1 and q2. Using new_q1 for now
            NDArray statesNewActions = states.concat(newActions, -1).toType(DataType.FLOAT32, false);
            NDArray newQ1 = this.qf1.getPredictor().predict(new NDList(statesNewActions)).singletonOrThrow();
            NDArray newQ = newQ1;

            // Policy KL loss: - (E_a[Q(s, a) + H(.)])
            NDArray policyKlLoss = newQ.sub(alpha.mul(newLogPi)).add(policyPriorLogProb).mean();
            try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                collector.backward(policyKlLoss);
                for (Pair<String, Parameter> params : policyModel.getModel().getBlock().getParameters()) {
                    NDArray paramsArr = params.getValue().getArray();
                    policyModel.getOptimizer().update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                }
            }

            // =========== Entropy Adjustment Step ===========

            // NOTE: In formula is alphas and not log_alphas
            NDArray alphasLoss = this.logAlpha.mul(newLogPi.squeeze(-1).add(this.tgtEntro).mean()).neg();
            NDArray hiuAlphasLoss = alphasLoss.sum();
            try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                collector.backward(hiuAlphasLoss);
                alphasOptimizer.update(this.logAlpha.getUid(), this.logAlpha, this.logAlpha.getGradient().duplicate());
            }
            this.logAlpha.clip(Math.log(SACParameter.MIN_ALPHA), Math.log(SACParameter.MAX_ALPHA));

            // =========== Target Networks Update Step ===========

            Helper.softParamUpdateFromTo(this.qf1, this.targetQf1, CommonParameter.SOFT_TARGET_TAU);
            Helper.softParamUpdateFromTo(this.qf2, this.targetQf2, CommonParameter.SOFT_TARGET_TAU);

        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }
}
