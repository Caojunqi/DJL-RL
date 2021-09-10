package agent.model;

import ai.djl.ndarray.NDManager;
import ai.djl.nn.AbstractBlock;

/**
 * @author Caojunqi
 * @date 2021-09-10 14:31
 */
public abstract class BaseModel extends AbstractBlock {
    private static final byte VERSION = 2;
    private final NDManager manager;

    public BaseModel(NDManager manager) {
        super(VERSION);
        this.manager = manager;
    }

    public NDManager getManager() {
        return manager;
    }
}
