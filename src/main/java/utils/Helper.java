package utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.util.Pair;
import algorithm.BaseModel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * 通用工具类
 *
 * @author Caojunqi
 * @date 2021-09-11 12:22
 */
public final class Helper {

    public static NDArray gather(NDArray arr, int[] indexes) {
        boolean[][] mask = new boolean[(int) arr.size(0)][(int) arr.size(1)];
        for (int i = 0; i < indexes.length; i++) {
            mask[i][indexes[i]] = true;
        }
        NDArray booleanMask = arr.getManager().create(mask);
        for (int i = booleanMask.getShape().dimension(); i < arr.getShape().dimension(); i++) {
            booleanMask = booleanMask.expandDims(i);
        }

        return arr.get(tile(booleanMask, arr.getShape())).reshape(Shape.update(arr.getShape(), 1, 1)).squeeze();
    }

    public static NDArray tile(NDArray arr, Shape shape) {
        for (int i = arr.getShape().dimension(); i < shape.dimension(); i++) {
            arr = arr.expandDims(i);
        }
        return arr.broadcast(shape);
    }

    public static double betweenDouble(double min, double max) {
        // 参数检查
        if (min > max) {
            throw new IllegalArgumentException("最小值[" + min + "]不能大于最大值[" + max + "]");
        }
        if (min == max) {
            return min;
        }
        return min + ThreadLocalRandom.current().nextDouble(max - min);
    }

    public static void shuffleArray(int[] array) {
        List<Integer> list = new ArrayList<>();
        for (int i : array) {
            list.add(i);
        }

        Collections.shuffle(list);

        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i);
        }
    }

    public static void copyModel(BaseModel fromModel, BaseModel toModel) {
        ParameterList fromParameterList = fromModel.getModel().getBlock().getParameters();
        ParameterList toParameterList = toModel.getModel().getBlock().getParameters();
        for (Pair<String, Parameter> params : toParameterList) {
            NDArray fromParam = fromParameterList.get(params.getKey()).getArray().duplicate();
            params.getValue().getArray().set(fromParam.toFloatArray());
        }
    }

    /**
     * 使用 soft update的办法来更新模型参数
     *
     * @param source 提供新的参数值来更新目标模型的参数
     * @param target 待更新的目标模型参数值
     * @param tau    参数更新比例
     */
    public static void softParamUpdateFromTo(BaseModel source, BaseModel target, double tau) {
        ParameterList sourceParameterList = source.getModel().getBlock().getParameters();
        ParameterList targetParameterList = target.getModel().getBlock().getParameters();
        for (Pair<String, Parameter> params : targetParameterList) {
            NDArray targetParam = params.getValue().getArray().duplicate();
            NDArray sourceParam = sourceParameterList.get(params.getKey()).getArray().duplicate();
            NDArray newTargetParam = targetParam.mul(1.0 - tau).add(sourceParam.mul(tau));
            params.getValue().getArray().set(newTargetParam.toFloatArray());
        }
    }
}
