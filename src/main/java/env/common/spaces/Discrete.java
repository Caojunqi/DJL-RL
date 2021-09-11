package env.common.spaces;

/**
 * 离散型空间
 * A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
 *
 * @author Caojunqi
 * @date 2021-09-11 16:04
 */
public class Discrete implements Space {
    /**
     * 离散型元素数目
     */
    private int num;

    public Discrete(int num) {
        this.num = num;
    }

    @Override
    public boolean contains(Object x) {
        if (!(x instanceof Integer)) {
            return false;
        }
        int xInt = (Integer) x;
        return xInt >= 0 && xInt < num;
    }

    @Override
    public int getDim() {
        return num;
    }
}
