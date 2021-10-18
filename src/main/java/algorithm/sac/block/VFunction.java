package algorithm.sac.block;

/**
 * State Value Function
 *
 * @author Caojunqi
 * @date 2021-10-12 15:25
 */
public class VFunction extends MLP {

    public VFunction(int[] hiddenSizes) {
        super(hiddenSizes, 1);
    }
}
