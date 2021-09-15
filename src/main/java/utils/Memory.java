package utils;

import ai.djl.ndarray.NDManager;
import env.common.action.Action;
import env.common.action.IActionCollector;
import org.apache.commons.lang3.Validate;
import utils.datatype.Transition;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 样本数据缓存
 *
 * @author Caojunqi
 * @date 2021-09-10 11:40
 */
public class Memory<A extends Action> {

    /**
     * 样本数据
     */
    private List<Transition<A>> transitions;

    public Memory() {
        this.transitions = new ArrayList<>();
    }

    public void addTransition(float[] state, A action, boolean done, float[] nextState, float reward) {
        Transition<A> transition = new Transition<>(state, action, done, nextState, reward);
        addTransition(transition);
    }

    public void addTransition(Transition<A> transition) {
        transitions.add(transition);
    }

    /**
     * 将缓存的样本数据随机打乱，用作采样数据
     *
     * @param manager 用来管理NDArray的生成
     * @return 采样结果
     */
    public MemoryBatch sample(NDManager manager) {
        List<Transition<A>> tmpList = new ArrayList<>(transitions);
        Collections.shuffle(tmpList);
        int batchSize = tmpList.size();
        Validate.isTrue(batchSize > 0, "采样异常，当前缓存样本数量为0！！");

        float[][] states = new float[batchSize][];
        IActionCollector actionCollector = null;
        boolean[] masks = new boolean[batchSize];
        float[][] nextStates = new float[batchSize][];
        float[] rewards = new float[batchSize];

        for (int i = 0; i < batchSize; i++) {
            Transition<A> transition = tmpList.get(i);
            states[i] = transition.getState();
            if (actionCollector == null) {
                try {
                    Class<?> collectorClz = transition.getAction().getCollectorClz();
                    Constructor<?> constructor = collectorClz.getConstructor();
                    actionCollector = (IActionCollector) constructor.newInstance();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
            actionCollector.addAction(transition.getAction());
            masks[i] = transition.isMasked();
            nextStates[i] = transition.getNextState();
            rewards[i] = transition.getReward();
        }
        return new MemoryBatch(manager.create(states), actionCollector.createNDArray(manager), manager.create(masks), manager.create(nextStates), manager.create(rewards));
    }

    public int getSize() {
        return transitions.size();
    }
}
