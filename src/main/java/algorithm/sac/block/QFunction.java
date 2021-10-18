package algorithm.sac.block;

/**
 * State-Action Value Function modeled with a MLP.
 *
 * @author Caojunqi
 * @date 2021-10-12 15:24
 */
public class QFunction extends MLP {

    public QFunction(int[] hiddenSizes) {
        super(hiddenSizes, 1);
    }
}
