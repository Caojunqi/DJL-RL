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
        NDArray dd = manager.create(new double[]{1, 0, 1, 0, 0, 1});
        NDArray b = dd.toType(DataType.BOOLEAN, true);

        NDArray bb = manager.create(new boolean[]{true, true, false, false, true});
        NDArray d = bb.toType(DataType.BOOLEAN, false);
        System.out.println("llll");
    }

}
