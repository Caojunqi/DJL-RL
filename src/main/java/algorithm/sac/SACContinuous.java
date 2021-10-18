package algorithm.sac;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import algorithm.BaseAlgorithm;
import algorithm.BaseModel;
import algorithm.ppo.model.BasePolicyModel;
import algorithm.sac.model.GaussianPolicyModel;
import algorithm.sac.model.QFunctionModel;
import env.common.action.impl.BoxAction;
import utils.ActionSampler;
import utils.MemoryBatch;

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
        try (NDManager subManager = manager.newSubManager()) {
            NDList distribution = policyModel.getPredictor().predict(new NDList(subManager.create(states)));
            double[] actionData = ActionSampler.sampleNormal(distribution.get(0), distribution.get(2), random);
            return new BoxAction(actionData);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public BoxAction greedyAction(float[] state) {
        // 此处将单一状态数组转为多维的，这样可以保证在predict过程中，传入1个状态和传入多个状态，输入数据的维度是一致的。
        float[][] states = new float[][]{state};
        try (NDManager subManager = manager.newSubManager()) {
            NDList distribution = policyModel.getPredictor().predict(new NDList(subManager.create(states)));
            double[] actionData = ActionSampler.sampleNormalGreedy(distribution.get(0));
            return new BoxAction(actionData);
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
            NDArray nextStates = batch.getNextStates();
            NDArray rewards = batch.getRewards();
            NDArray masks = batch.getMasks();

            float policyPriorLogProb = 0.0f; // Uniform prior // TODO: Normal prior

            // Alphas
            NDArray alpha = this.entropyScale.mul(this.logAlpha.exp());

//        } catch (TranslateException e) {
//            throw new IllegalStateException(e);
        }
    }
}
