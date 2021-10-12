package algorithm;

import ai.djl.nn.AbstractBlock;

/**
 * @author Caojunqi
 * @date 2021-09-10 14:31
 */
public abstract class BaseModelBlock extends AbstractBlock {
    private static final byte VERSION = 2;

    public BaseModelBlock() {
        super(VERSION);
    }
}
