package env.common.spaces.state;

/**
 * 离散型状态空间
 *
 * @author Caojunqi
 * @date 2021-09-13 12:20
 */
public class DiscreteStateSpace implements StateSpace {

    /**
     * 离散型元素数目
     */
    private int num;

    public DiscreteStateSpace(int num) {
        this.num = num;
    }

    @Override
    public int getDim() {
        return num;
    }
}
