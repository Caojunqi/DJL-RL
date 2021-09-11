package env.common.spaces;

/**
 * Defines the observation and action spaces, so you can write generic
 * code that applies to any Env. For example, you can choose a random
 * action.
 *
 * @author Caojunqi
 * @date 2021-09-11 15:38
 */
public interface Space {

    /**
     * Return boolean specifying if x is a valid
     * member of this space
     */
    boolean contains(Object x);

    /**
     * Return the number of space dimensions
     */
    int getDim();
}
