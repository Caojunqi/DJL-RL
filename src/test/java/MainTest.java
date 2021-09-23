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
        NDArray actions = manager.create(new float[][]{
                {1.0692f},
                {0.8564f},
                {0.6943f}
        });
        NDArray actionMean = manager.create(new float[][]{
                {0.0120f},
                {0.0120f},
                {0.0120f}
        });
        NDArray actionLogStd = manager.create(new float[][]{
                {-0.2300f},
                {-0.2300f},
                {-0.2300f}
        });
        NDArray actionStd = manager.create(new float[][]{
                {0.7945f},
                {0.7945f},
                {0.7945f}
        });

        NDArray result = normalLogDensity(actions, actionMean, actionLogStd, actionStd);
        System.out.println("22222");

    }

    public static NDArray normalLogDensity(NDArray actions, NDArray actionMean, NDArray actionLogStd, NDArray actionStd) {
        NDArray var = actionStd.pow(2);
        NDArray logDensity = actions.sub(actionMean).pow(2).div(var.mul(2)).neg().sub(Math.log(2 * Math.PI) * 0.5).sub(actionLogStd);
        return logDensity.sum(new int[]{1}, true);
    }

}
