package algorithm.ppo2;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import algorithm.BaseModel;
import env.action.core.IAction;
import utils.datatype.PolicyPair;

/**
 * Policy class for actor-critic algorithms (has both policy and value prediction).
 *
 * @author Caojunqi
 * @date 2021-11-27 11:15
 */
public abstract class BaseActorCriticModel<A extends IAction> extends BaseModel {

    /**
     * 进行策略选择
     *
     * @param states           环境状态集合
     * @param deterministic    是否选择确定性策略，true-确定性策略，通常用于模型评估阶段；false-随机性策略，通常用于模型训练阶段。
     * @param returnPolicyInfo 是否返回策略选择过程中所使用的的其他数据信息，例如“所选动作数组”、“动作均值数组”、“动作方差数组”、“动作方差对数数组”、“所选动作似然度对数，也即计算出策略π的对数 ∑ log π(a_t | s_t)”
     * @param noGrad           true-模型推算出来的结果需要清除梯度信息；false-模型推算出来的结果需要保留梯度信息
     * @return 策略选择结果，包含策略选择的动作结果，以及其他信息
     */
    public abstract PolicyPair<A> policy(NDList states, boolean deterministic, boolean returnPolicyInfo, boolean noGrad);

    public abstract NDArray value(NDList states);
}
