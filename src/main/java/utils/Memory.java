package utils;

import ai.djl.ndarray.NDManager;
import utils.datatype.Transition;

import java.util.*;

/**
 * 样本数据缓存
 *
 * @author Caojunqi
 * @date 2021-09-10 11:40
 */
public class Memory {

    private Random random;
    /**
     * 缓存容量，最多可以缓存多少个样本数据
     */
    private int capacity;
    /**
     * 样本数据
     */
    private Transition[] transitions;
    /**
     * 当前缓存指针，表示接下来的样本会存放在{@link this#transitions}的哪个位置
     */
    private int head;
    /**
     * 已经缓存的样本个数
     */
    private int size;

    public Memory(int capacity) {
        this(capacity, 0);
    }

    public Memory(int capacity, int seed) {
        this.capacity = capacity;
        this.transitions = new Transition[capacity];
        this.random = new Random(seed);
    }

    public void addTransition(float[] state, int action, boolean done, float[] nextState, float reward) {
        Transition transition = new Transition(state, action, done, nextState, reward);
        addTransition(transition);
    }

    public void addTransition(Transition transition) {
        transitions[head] = transition;
        if (size < capacity) {
            size++;
        }

        head += 1;
        if (head >= capacity) {
            head = 0;
        }
    }

    /**
     * 将缓存的样本数据随机打乱，用作采样数据
     *
     * @param manager 用来管理NDArray的生成
     * @return 采样结果
     */
    public MemoryBatch sample(NDManager manager) {
        List<Transition> tmpList = new ArrayList<>(Arrays.asList(transitions));
        Collections.shuffle(tmpList);
        int batchSize = tmpList.size();

        float[][] states = new float[batchSize][];
        int[] actions = new int[batchSize];
        boolean[] masks = new boolean[batchSize];
        float[][] nextStates = new float[batchSize][];
        float[] rewards = new float[batchSize];

        for (int i = 0; i < batchSize; i++) {
            Transition transition = tmpList.get(i);
            states[i] = transition.getState();
            actions[i] = transition.getAction();
            masks[i] = transition.isMasked();
            nextStates[i] = transition.getNextState();
            rewards[i] = transition.getReward();
        }
        return new MemoryBatch(manager.create(states), manager.create(actions), manager.create(masks), manager.create(nextStates), manager.create(rewards));
    }

    public int getSize() {
        return size;
    }
}
