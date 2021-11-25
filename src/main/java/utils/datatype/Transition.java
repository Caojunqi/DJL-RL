package utils.datatype;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import env.action.core.IAction;
import env.state.core.IState;

import java.util.HashMap;
import java.util.Map;

/**
 * 一个可用的数据样本
 *
 * @author Caojunqi
 * @date 2021-09-10 11:44
 */
public class Transition<S extends IState, A extends IAction> {
    /**
     * 执行指定动作之前的环境状态
     */
    private S state;
    /**
     * 所执行的动作
     */
    private A action;
    /**
     * 执行指定动作后，当前幕是否结束
     */
    private boolean mask;
    /**
     * 执行指定动作后的环境状态
     */
    private S nextState;
    /**
     * 执行指定动作所获取的收益
     */
    private float reward;

    public Transition(S state, A action, boolean mask, S nextState, float reward) {
        this.state = state;
        this.action = action;
        this.mask = mask;
        this.nextState = nextState;
        this.reward = reward;
    }

    public S getState() {
        return state;
    }

    public A getAction() {
        return action;
    }

    public void setAction(A action) {
        this.action = action;
    }

    public boolean isMasked() {
        return mask;
    }

    public S getNextState() {
        return nextState;
    }

    public float getReward() {
        return reward;
    }

    @Override
    public String toString() {
        try {
            Map<String, Object> info = new HashMap<>();
            info.put("state", state);
            info.put("action", action);
            info.put("mask", mask);
            info.put("nextState", nextState);
            info.put("reward", reward);
            return new ObjectMapper().writeValueAsString(info);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }
}
