package utils.datatype;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.HashMap;
import java.util.Map;

/**
 * 环境快照
 *
 * @author Caojunqi
 * @date 2021-09-09 22:03
 */
public class Snapshot {
    /**
     * 环境状态信息
     */
    private float[] state;
    /**
     * 当前步收益
     */
    private float reward;
    /**
     * 当前幕是否结束，1-结束，0-未结束
     */
    private boolean done;

    public Snapshot(float[] state, float reward, boolean done) {
        this.state = state.clone();
        this.reward = reward;
        this.done = done;
    }

    public float[] getState() {
        return state;
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
            info.put("state", state);
            info.put("reward", reward);
            info.put("done", done);
            return new ObjectMapper().writeValueAsString(info);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }
}
