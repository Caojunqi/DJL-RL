package env.common.spaces.state;

import org.apache.commons.lang3.Validate;

/**
 * 多维度离散型状态空间
 *
 * @author Caojunqi
 * @date 2021-09-13 12:21
 */
public class MultiDiscreteStateSpace implements StateSpace {

    /**
     * 离散型数据的取值，该数组长度表示数据维度，数据值表示对应维度的离散数据取值最大值（不含）。
     */
    private int[] counts;

    public MultiDiscreteStateSpace(int[] counts) {
        Validate.isTrue(counts != null, "multi-discrete state space data is invalid!!");
        this.counts = counts;
    }

    @Override
    public int getDim() {
        return this.counts.length;
    }
}
