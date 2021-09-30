package model.model;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.optimizer.Optimizer;

import java.util.Random;

/**
 * 模型基类
 *
 * @author Caojunqi
 * @date 2021-09-23 21:59
 */
public abstract class BaseModel {
    protected Random random = new Random(0);
    protected NDManager manager;
    protected Optimizer optimizer;
    protected Model model;
    protected Predictor<NDList, NDList> predictor;

    public Optimizer getOptimizer() {
        return optimizer;
    }

    public Model getModel() {
        return model;
    }

    public Predictor<NDList, NDList> getPredictor() {
        return predictor;
    }
}
