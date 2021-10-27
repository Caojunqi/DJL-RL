package utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import org.apache.commons.lang3.Validate;

import java.util.Random;

/**
 * 动作采样器
 *
 * @author Caojunqi
 * @date 2021-09-10 18:20
 */
public final class ActionSampler {

    private static final double EPS = 1e-8;

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
     * 从离散型动作分布中随机抽取一个动作
     *
     * @param distribution 动作分布，其数据长度就是可选动作的数量，数据值表示该index的动作被选中的概率
     * @param random       随机数生成器
     * @return 选中的动作
     */
    public static NDArray sampleMultinomial(NDManager manager, NDArray distribution, Random random) {
        int sampleSize = (int) distribution.getShape().get(0);
        int actionSize = (int) distribution.getShape().get(1);
        int[] actionData = new int[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            int value = 0;
            float rnd = random.nextFloat();
            for (int j = 0; j < actionSize; j++) {
                float cut = distribution.getFloat(i, value);
                if (rnd > cut) {
                    value++;
                } else {
                    actionData[i] = value;
                }
                rnd -= cut;
            }
        }
        return manager.create(actionData);
    }

    /**
     * 从连续型动作分布中随机获取动作，按照正态分布来抽样
     *
     * @param actionMean 动作均值，其数据长度表示连续型动作的参数个数
     * @param actionStd  动作方差，其数据长度也表示连续型动作的参数个数，应和actionMean长度保持一致
     * @param random     随机数生成器
     * @return 随机选取的动作数据
     */
    public static double[] normalSampleActionData(NDArray actionMean, NDArray actionStd, Random random) {
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
     * 当采用正态分布策略来进行采样时，可使用该接口来计算采样结果的似然度对数，
     * 即：计算 log(π(a|s))
     *
     * @param action       采样结果
     * @param actionMean   动作平均值
     * @param actionStd    动作标准差
     * @param actionLogStd 动作标准差对数值
     * @return 似然度对数值
     */
    public static NDArray sampleLogProb(NDArray action, NDArray actionMean, NDArray actionStd, NDArray actionLogStd) {
        NDArray logProb = action.sub(actionMean).div(actionStd.add(EPS)).pow(2).add(actionLogStd.mul(2)).add(Math.log(2 * Math.PI));
        logProb.muli(-0.5);
        logProb = logProb.sum(new int[]{-1}, true);
        return logProb;
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
