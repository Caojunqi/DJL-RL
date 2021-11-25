package utils.datatype;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import env.state.core.IState;

import java.util.HashMap;
import java.util.Map;

/**
 * 当Agent与环境进行一次交互后，相关数据的快照
 *
 * @author Caojunqi
 * @date 2021-09-09 22:03
 */
public class Snapshot<S extends IState> {
    /**
     * Agent执行特定行为后，环境的状态
     */
    private S nextState;
    /**
     * Agent执行特定行为后，获得的收益
     */
    private float reward;
    /**
     * Agent执行特定行为后，当前幕是否结束，1-结束，0-未结束
     */
    private boolean done;

    public Snapshot(S nextState, float reward, boolean done) {
        this.nextState = nextState;
        this.reward = reward;
        this.done = done;
    }

    public S getNextState() {
        return nextState;
    }

    public float getReward() {
        return reward;
    }

    public boolean isDone() {
        return done;
    }

    @Override
    public String toString() {
        try {
            Map<String, Object> info = new HashMap<>();
            info.put("nextState", nextState);
            info.put("reward", reward);
            info.put("done", done);
            return new ObjectMapper().writeValueAsString(info);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }
}
