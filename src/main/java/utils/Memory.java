package utils;

import utils.datatype.Transition;

import java.util.Random;

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
     * 随机采样指定数量的样本
     *
     * @param sampleSize 所需样本数量
     * @return 采样结果
     */
    public Transition[] sample(int sampleSize) {
        Transition[] chunk = new Transition[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            chunk[i] = transitions[random.nextInt(size)];
        }
        return chunk;
    }

    public int getSize() {
        return size;
    }
}
