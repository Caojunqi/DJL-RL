package model.model;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.training.optimizer.Optimizer;

import java.util.Random;

/**
 * 策略模型基类
 *
 * @author Caojunqi
 * @date 2021-09-23 22:00
 */
public abstract class BasePolicyModel extends BaseModel {
    protected Random random = new Random(0);
    protected Optimizer optimizer;
    protected Model model;
    protected Predictor<NDList, NDList> predictor;
}
