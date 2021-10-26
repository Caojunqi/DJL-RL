import ai.djl.ndarray.NDManager;
import algorithm.BaseModel;
import algorithm.sac.model.QFunctionModel;
import utils.Helper;

/**
 * 测试类
 *
 * @author Caojunqi
 * @date 2021-09-10 10:06
 */
public class MainTest {

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();
        BaseModel qf1 = QFunctionModel.newModel(manager, 3, 2);
        BaseModel qf2 = QFunctionModel.newModel(manager, 3, 2);
        Helper.copyModel(qf1, qf2);
        System.out.println("llll");
    }

}
