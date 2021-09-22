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
        NDManager manager = NDManager.newBaseManager();
        NDArray nd = manager.create(new float[][]{{0.9896f}, {0.0204f}, {-1.0100f}});
        NDArray mean = nd.mean();
        System.out.println("ccccc");
    }

}
