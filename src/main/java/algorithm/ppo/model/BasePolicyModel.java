package algorithm.ppo.model;

import ai.djl.ndarray.NDList;
import algorithm.BaseModel;
import env.common.action.Action;
import utils.datatype.PolicyPair;

import java.util.Random;

/**
 * 策略模型基类
 *
 * @author Caojunqi
 * @date 2021-09-23 22:00
 */
public abstract class BasePolicyModel<A extends Action> extends BaseModel {
    /**
     * 随机数生成器
     */
    protected Random random = new Random(0);

    /**
     * 进行策略选择
     *
     * @param states        环境状态集合
     * @param deterministic 是否选择确定性策略，true-确定性策略，通常用于模型评估阶段；false-随机性策略，通常用于模型训练阶段。
     * @param returnLogProb 使用策略计算出动作后，是否返回动作似然度对数，也即计算出策略π的对数 ∑ log π(a_t | s_t)
     * @return 策略选择结果，包含策略选择的动作结果，以及其他信息，例如 “连续型动作均值”、“连续型动作方差”、“动作似然度对数”等，其他信息的个数不固定
     */
    public abstract PolicyPair<A> policy(NDList states, boolean deterministic, boolean returnLogProb);

}
