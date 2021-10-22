import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;

/**
 * 测试类
 *
 * @author Caojunqi
 * @date 2021-09-10 10:06
 */
public class MainTest {

    public static void main(String[] args) {
        int actionDim = 2;
        NDManager manager = NDManager.newBaseManager();
        NDArray dd = manager.create(new float[]{1.2f, 0.3f, 1.3f, 0.4f, 0.9f, 1.7f});
        double[] ddd = dd.toType(DataType.FLOAT64, true).toDoubleArray();
        System.out.println("llll");
    }

}
