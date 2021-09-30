package utils;

import ai.djl.ndarray.NDArray;
import org.apache.commons.lang3.Validate;

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

    /**
     * 从离散型动作分布中随机抽取一个动作
     *
     * @param distribution 动作分布，其数据长度就是可选动作的数量，数据值表示该index的动作被选中的概率
     * @param random       随机数生成器
     * @return 选中的动作
     */
    public static int sampleMultinomial(NDArray distribution, Random random) {
        // 剔除掉多余的维度
        NDArray squeezeDistribution = distribution.squeeze();
        int value = 0;
        long size = squeezeDistribution.size();
        float rnd = random.nextFloat();
        for (int i = 0; i < size; i++) {
            float cut = squeezeDistribution.getFloat(value);
            if (rnd > cut) {
                value++;
            } else {
                return value;
            }
            rnd -= cut;
        }

        throw new IllegalArgumentException("Invalid multinomial distribution");
    }

    /**
     * 从连续型动作分布中随机获取动作，按照正态分布来抽样
     *
     * @param actionMean 动作均值，其数据长度表示连续型动作的参数个数
     * @param actionStd  动作方差，其数据长度也表示连续型动作的参数个数，应和actionMean长度保持一致
     * @param random     随机数生成器
     * @return 随机选取的动作数据
     */
    public static double[] sampleNormal(NDArray actionMean, NDArray actionStd, Random random) {
        NDArray squeezeActionMean = actionMean.squeeze(0);
        NDArray squeezeActionStd = actionStd.squeeze(0);
        Validate.isTrue(squeezeActionMean.size() == squeezeActionStd.size(), "随机抽样一个连续型动作数据时，动作均值和动作方差的数据长度应该一致！！");
        int parameterSize = (int) squeezeActionMean.size();
        double[] actionData = new double[parameterSize];
        for (int i = 0; i < parameterSize; i++) {
            float mean = squeezeActionMean.getFloat(i);
            float std = squeezeActionStd.getFloat(i);
            actionData[i] = random.nextGaussian() * std + mean;
        }
        return actionData;
    }

    /**
     * 从连续型动作分布中按照贪婪策略获取动作，按照正态分布来抽样
     *
     * @param actionMean 动作均值，其数据长度表示连续型动作的参数个数
     * @return 贪婪策略选取的动作数据
     */
    public static double[] sampleNormalGreedy(NDArray actionMean) {
        NDArray squeezeActionMean = actionMean.squeeze(0);
        int parameterSize = (int) squeezeActionMean.size();
        double[] actionData = new double[parameterSize];
        for (int i = 0; i < parameterSize; i++) {
            float mean = squeezeActionMean.getFloat(i);
            actionData[i] = mean;
        }
        return actionData;
    }
}
