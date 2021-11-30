package algorithm.td3;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
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
import algorithm.sac.SACParameter;
import algorithm.sac.model.QFunctionModel;
import algorithm.td3.model.ActorModel;
import env.action.core.impl.BoxAction;
import env.common.Environment;
import env.state.core.IState;
import utils.Helper;
import utils.MemoryBatch;
import utils.datatype.PolicyPair;

import java.util.Arrays;

/**
 * TD3算法 Twin Delayed Deep Deterministic
 * 该算法仅适用于连续型动作空间
 *
 * @author Caojunqi
 * @date 2021-11-01 11:01
 */
public class TD3Continuous<S extends IState<S>> extends BaseAlgorithm<S, BoxAction> {
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
     * 目标策略模型
     */
    private BasePolicyModel<BoxAction> targetPolicyModel;
    /**
     * 目标Q函数模型_1
     */
    private BaseModel targetQf1;
    /**
     * 目标Q函数模型_2
     */
    private BaseModel targetQf2;

    public TD3Continuous(NDManager manager, Environment<S, BoxAction> env) {
        super(manager);
        int stateDim = env.getStateSpaceDim();
        int actionDim = env.getActionSpaceDim();
        this.policyModel = ActorModel.newModel(manager, stateDim, actionDim);
        this.policyOptimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(TD3Parameter.POLICY_LR)).optWeightDecays(SACParameter.POLICY_WEIGHT_DECAY).build();
        this.qf1 = QFunctionModel.newModel(manager, stateDim, actionDim);
        this.qfOptimizer1 = Optimizer.adam().optLearningRateTracker(Tracker.fixed(TD3Parameter.QF_LR)).build();
        this.qf2 = QFunctionModel.newModel(manager, stateDim, actionDim);
        this.qfOptimizer2 = Optimizer.adam().optLearningRateTracker(Tracker.fixed(TD3Parameter.QF_LR)).build();
        this.targetPolicyModel = ActorModel.newModel(manager, stateDim, actionDim);
        this.targetQf1 = QFunctionModel.newModel(manager, stateDim, actionDim);
        this.targetQf2 = QFunctionModel.newModel(manager, stateDim, actionDim);

        Helper.copyModel(policyModel, targetPolicyModel);
        Helper.copyModel(qf1, targetQf1);
        Helper.copyModel(qf2, targetQf2);
    }

    @Override
    public BoxAction selectAction(S state) {
        NDManager subManager = manager.newSubManager();
        return policyModel.policy(state.singleStateList(subManager), false, false, true).singletonOrThrow();
    }

    @Override
    public BoxAction greedyAction(S state) {
        NDManager subManager = manager.newSubManager();
        return policyModel.policy(state.singleStateList(subManager), true, false, true).singletonOrThrow();
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

                    PolicyPair<BoxAction> nextPolicyPair = this.targetPolicyModel.policy(new NDList(nextStatesSubset), false, true, true);
                    NDArray nextActions = nextPolicyPair.getInfo().get(0);

                    NDArray nextStatesActions = nextStatesSubset.concat(nextActions, -1).toType(DataType.FLOAT32, false);
                    NDArray nextQ1 = this.targetQf1.getPredictor().predict(new NDList(nextStatesActions)).singletonOrThrow().duplicate();
                    NDArray nextQ2 = this.targetQf2.getPredictor().predict(new NDList(nextStatesActions)).singletonOrThrow().duplicate();
                    NDArray nextQ = nextQ1.minimum(nextQ2);

                    NDArray qBackup = rewardsSubset.add(terminationsSubset.neg().add(1).mul(CommonParameter.GAMMA).mul(nextQ));

                    NDArray statesActions = statesSubset.concat(actionsSubset, -1).toType(DataType.FLOAT32, false);
                    NDArray predQ1 = this.qf1.getPredictor().predict(new NDList(statesActions)).singletonOrThrow();
                    NDArray lossQf1 = predQ1.sub(qBackup).pow(2).mean().mul(0.5).squeeze(-1);

                    NDArray predQ2 = this.qf2.getPredictor().predict(new NDList(statesActions)).singletonOrThrow();
                    NDArray lossQf2 = predQ2.sub(qBackup).pow(2).mean().mul(0.5).squeeze(-1);

                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(lossQf1);
                        for (Pair<String, Parameter> params : qf1.getModel().getBlock().getParameters()) {
                            NDArray paramsArr = params.getValue().getArray();
                            qfOptimizer1.update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                        }
                    }

                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(lossQf2);
                        for (Pair<String, Parameter> params : qf2.getModel().getBlock().getParameters()) {
                            NDArray paramsArr = params.getValue().getArray();
                            qfOptimizer2.update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                        }
                    }

                    Helper.softParamUpdateFromTo(this.qf1, this.targetQf1, CommonParameter.SOFT_TARGET_TAU);
                    Helper.softParamUpdateFromTo(this.qf2, this.targetQf2, CommonParameter.SOFT_TARGET_TAU);

                    // TODO 在更新policy参数前，有一步定期更新学习率的操作

                    for (Pair<String, Parameter> params : qf1.getModel().getBlock().getParameters()) {
                        params.getValue().getArray().setRequiresGradient(false);
                    }

                    PolicyPair<BoxAction> newPolicyPair = this.policyModel.policy(new NDList(statesSubset), true, true, false);
                    NDArray newActions = newPolicyPair.getInfo().get(0);
                    NDArray statesNewActions = statesSubset.concat(newActions, -1).toType(DataType.FLOAT32, false);
                    NDArray newQ1 = this.qf1.getPredictor().predict(new NDList(statesNewActions)).singletonOrThrow();
                    NDArray actorLoss = newQ1.neg().mean();
                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(actorLoss);
                        for (Pair<String, Parameter> params : policyModel.getModel().getBlock().getParameters()) {
                            NDArray paramsArr = params.getValue().getArray();
                            policyOptimizer.update(params.getKey(), paramsArr, paramsArr.getGradient().duplicate());
                        }
                    }
                    Helper.softParamUpdateFromTo(this.policyModel, this.targetPolicyModel, CommonParameter.SOFT_TARGET_TAU);

                    for (Pair<String, Parameter> params : qf1.getModel().getBlock().getParameters()) {
                        params.getValue().getArray().setRequiresGradient(true);
                    }
                }
            }
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }
}
