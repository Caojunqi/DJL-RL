import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * 测试类
 *
 * @author Caojunqi
 * @date 2021-09-10 10:06
 */
public class MainTest {

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();
        NDArray array = manager.randomInteger(1, 10, new Shape(30, 2, 3, 4), DataType.INT32);
        NDArray newArray = array.reshape(array.getShape().get(0), -1);
        System.out.println("llll");
    }

}
