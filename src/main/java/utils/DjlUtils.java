package utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

/**
 * DJL框架扩展相关的工具方法
 *
 * @author Caojunqi
 * @date 2021-11-26 10:37
 */
public final class DjlUtils {

    /**
     * 根据指定的三维float数据创建对应的NDArray
     */
//    public static NDArray create(NDManager manager, float[][][] data) {
//        FloatBuffer buffer = FloatBuffer.allocate(data.length * data[0].length * data[0][0].length);
//        for (float[][] dd : data) {
//            for (float[] d : dd) {
//                buffer.put(d);
//            }
//        }
//        buffer.rewind();
//        return manager.create(buffer, new Shape(data.length, data[0].length, data[0][0].length));
//    }

    /**
     * 根据指定的三维float数据创建对应的NDArray
     */
    public static NDArray create(NDManager manager, float[][][] data) {
        int d1 = data.length;
        int d2 = data[0].length;
        int d3 = data[0][0].length;

        float[] oneDimData = new float[d1 * d2 * d3];
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int h = 0; h < d3; h++) {
                    int index = (i * d2 + j) * d3 + h;
                    oneDimData[index] = data[i][j][h];
                }
            }
        }
        return manager.create(oneDimData, new Shape(d1, d2, d3));
    }

    /**
     * 根据指定的四维float数据创建对应的NDArray
     */
    public static NDArray create(NDManager manager, float[][][][] data) {
        int d1 = data.length;
        int d2 = data[0].length;
        int d3 = data[0][0].length;
        int d4 = data[0][0][0].length;

        float[] oneDimData = new float[d1 * d2 * d3 * d4];
        for (int i = 0; i < d1; i++) {
            for (int j = 0; j < d2; j++) {
                for (int h = 0; h < d3; h++) {
                    for (int k = 0; k < d4; k++) {
                        int index = ((i * d2 + j) * d3 + h) * d4 + k;
                        oneDimData[index] = data[i][j][h][k];
                    }
                }
            }
        }
        return manager.create(oneDimData, new Shape(d1, d2, d3, d4));
    }
}
