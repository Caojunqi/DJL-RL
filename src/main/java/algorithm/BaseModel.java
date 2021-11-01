package algorithm;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;

/**
 * 模型基类
 *
 * @author Caojunqi
 * @date 2021-09-23 21:59
 */
public abstract class BaseModel {
    protected NDManager manager;
    protected Model model;
    protected Predictor<NDList, NDList> predictor;

    public Model getModel() {
        return model;
    }

    public Predictor<NDList, NDList> getPredictor() {
        return predictor;
    }
}
