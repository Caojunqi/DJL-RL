package utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

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

    /**
     * 将指定Object对象转为List对象
     *
     * @param obj 待转化的对象
     * @return 转化结果
     */
    public static List<Object> convertObjectToList(Object obj) {
        List<Object> list = new ArrayList<>();
        if (obj.getClass().isArray()) {
            list = Arrays.asList((Object[]) obj);
        } else if (obj instanceof Collection) {
            list = new ArrayList<>((Collection<?>) obj);
        }
        return list;
    }
}
