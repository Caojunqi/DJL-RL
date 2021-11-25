package utils;

import ai.djl.ndarray.NDManager;
import env.action.collector.IActionCollector;
import env.action.core.IAction;
import env.state.collector.IStateCollector;
import env.state.core.IState;
import org.apache.commons.lang3.Validate;
import utils.datatype.Transition;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;

/**
 * 样本数据缓存
 *
 * @author Caojunqi
 * @date 2021-09-10 11:40
 */
public class Memory<S extends IState, A extends IAction> {

    /**
     * 样本数据
     */
    private List<Transition<S, A>> transitions;

    public Memory() {
        this.transitions = new ArrayList<>();
    }

    public void addTransition(S state, A action, boolean done, S nextState, float reward) {
        Transition<S, A> transition = new Transition<>(state, action, done, nextState, reward);
        addTransition(transition);
    }

    public void addTransition(Transition<S, A> transition) {
        transitions.add(transition);
    }

    public void reset() {
        this.transitions.clear();
    }

    /**
     * 将缓存的样本数据随机打乱，用作采样数据
     *
     * @param manager 用来管理NDArray的生成
     * @return 采样结果
     */
    public MemoryBatch sample(NDManager manager) {
        List<Transition<S, A>> tmpList = new ArrayList<>(transitions);
//        Collections.shuffle(tmpList);
        int batchSize = tmpList.size();
        Validate.isTrue(batchSize > 0, "采样异常，当前缓存样本数量为0！！");

        IStateCollector stateCollector = null;
        IActionCollector actionCollector = null;
        boolean[][] masks = new boolean[batchSize][];
        IStateCollector nextStateCollector = null;
        float[][] rewards = new float[batchSize][];

        for (int i = 0; i < batchSize; i++) {
            Transition<S, A> transition = tmpList.get(i);
            if (stateCollector == null) {
                try {
                    Class<?> collectorClz = transition.getState().getCollectorClz();
                    Constructor<?> constructor = collectorClz.getConstructor(int.class);
                    stateCollector = (IStateCollector) constructor.newInstance(batchSize);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
            if (actionCollector == null) {
                try {
                    Class<?> collectorClz = transition.getAction().getCollectorClz();
                    Constructor<?> constructor = collectorClz.getConstructor(int.class);
                    actionCollector = (IActionCollector) constructor.newInstance(batchSize);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
            if (nextStateCollector == null) {
                try {
                    Class<?> collectorClz = transition.getNextState().getCollectorClz();
                    Constructor<?> constructor = collectorClz.getConstructor(int.class);
                    nextStateCollector = (IStateCollector) constructor.newInstance(batchSize);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
            stateCollector.addState(i, transition.getState());
            actionCollector.addAction(i, transition.getAction());
            masks[i] = new boolean[]{transition.isMasked()};
            nextStateCollector.addState(i, transition.getNextState());
            rewards[i] = new float[]{transition.getReward()};
        }
        return new MemoryBatch(stateCollector.createNDArray(manager),
                actionCollector.createNDArray(manager),
                manager.create(masks),
                nextStateCollector.createNDArray(manager),
                manager.create(rewards));
    }

    public int getSize() {
        return transitions.size();
    }
}
