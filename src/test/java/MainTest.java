import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

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
        NDArray array = manager.create(new double[]{0, Math.PI});
        NDArray tanh = array.tanh();
        NDArray tan = array.tan();
        System.out.println("llll");
    }

}
