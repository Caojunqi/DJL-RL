package utils;

import ai.djl.ndarray.NDArray;

import java.util.Random;

/**
 * 动作采样器
 *
 * @author Caojunqi
 * @date 2021-09-10 18:20
 */
public final class ActionSampler {

    public static int epsilonGreedy(NDArray distribution, Random random, float epsilon) {
        if (random.nextFloat() < epsilon) {
            return random.nextInt((int) distribution.size());
        } else {
            return greedy(distribution);
        }
    }

    public static int greedy(NDArray distribution) {
        return (int) distribution.argMax().getLong();
    }

    public static int sampleMultinomial(NDArray distribution, Random random) {
        int value = 0;
        long size = distribution.size();
        float rnd = random.nextFloat();
        for (int i = 0; i < size; i++) {
            float cut = distribution.getFloat(value);
            if (rnd > cut) {
                value++;
            } else {
                return value;
            }
            rnd -= cut;
        }

        throw new IllegalArgumentException("Invalid multinomial distribution");
    }
}
