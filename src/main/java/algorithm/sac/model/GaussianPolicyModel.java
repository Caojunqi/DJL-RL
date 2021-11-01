package algorithm.sac.model;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import algorithm.BaseModelBlock;
import algorithm.ppo.model.BasePolicyModel;
import algorithm.sac.SACParameter;
import algorithm.sac.block.GaussianPolicy;
import env.common.action.impl.BoxAction;
import utils.ActionSampler;
import utils.datatype.PolicyPair;

import java.util.ArrayList;
import java.util.List;

/**
 * Gaussian策略模型
 *
 * @author Caojunqi
 * @date 2021-10-12 17:09
 */
public class GaussianPolicyModel extends BasePolicyModel<BoxAction> {

    private GaussianPolicyModel() {
        // 私有化构造器
    }

    public static GaussianPolicyModel newModel(NDManager manager, int stateDim, int actionDim) {
        Model model = Model.newInstance("gaussian_policy_model");
        BaseModelBlock net = new GaussianPolicy(SACParameter.NETS_HIDDEN_SIZES, actionDim);
        net.initialize(manager, DataType.FLOAT32, new Shape(stateDim));
        model.setBlock(net);

        GaussianPolicyModel gaussianPolicyModel = new GaussianPolicyModel();
        gaussianPolicyModel.manager = manager;
        gaussianPolicyModel.model = model;
        gaussianPolicyModel.predictor = model.newPredictor(new NoopTranslator());
        return gaussianPolicyModel;
    }

    @Override
    public PolicyPair<BoxAction> policy(NDList states, boolean deterministic, boolean returnPolicyInfo, boolean noGrad) {
        try (NDManager subManager = manager.newSubManager()) {
            NDList distribution = predictor.predict(states);
            NDArray mean = distribution.get(0);
            NDArray logStd = distribution.get(1);
            NDArray std = distribution.get(2);
            if (noGrad) {
                mean = mean.duplicate();
                logStd = logStd.duplicate();
                std = std.duplicate();
            }

            NDArray action;
            if (deterministic) {
                action = mean;
            } else {
                action = normalSampleActionArray(subManager, mean, std);
            }

            // Action between -1 and 1
            NDArray actionTanh = action.tanh();

            NDList info = null;
            if (returnPolicyInfo) {
                NDArray logProb = ActionSampler.sampleLogProb(action, mean, std, logStd);
                if (!deterministic) {
                    // 由于Action结果被tanh函数压缩到了(-1,1)，所以logProb的计算过程也要进行相应的调整
                    // 此处的公式参看论文"Soft Actor-Critic:Off-Policy Maximum Entropy DeepReinforcement Learning with a Stochastic Actor"的附录"C. Enforcing Action Bounds"
                    logProb.subi(actionTanh.pow(2).neg().add(1).clip(0, 1).add(1.e-6).log());
                    logProb = logProb.sum(new int[]{-1}, true);
                }
                info = new NDList(actionTanh, mean, std, logStd, logProb);
            }

            int actionSize = (int) actionTanh.getShape().get(0);
            List<BoxAction> actions = new ArrayList<>();
            for (int i = 0; i < actionSize; i++) {
                float[] actionData = actionTanh.get(new NDIndex(i + ",:")).toFloatArray();
                actions.add(new BoxAction(actionData));
            }

            return PolicyPair.of(actions, info);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * 从连续型动作分布中随机获取动作，按照正态分布来抽样
     *
     * @param actionMean 动作均值，其数据长度表示连续型动作的参数个数
     * @param actionStd  动作方差，其数据长度也表示连续型动作的参数个数，应和actionMean长度保持一致
     * @return 随机选取的动作数据
     */
    public NDArray normalSampleActionArray(NDManager manager, NDArray actionMean, NDArray actionStd) {
        NDArray noise = manager.randomNormal(actionStd.getShape());
        return actionStd.mul(noise).add(actionMean);
    }
}
