package env.common.spaces.state;

import org.apache.commons.lang3.Validate;

/**
 * 连续型状态空间
 *
 * @author Caojunqi
 * @date 2021-09-13 12:11
 */
public class BoxStateSpace implements StateSpace {

    /**
     * 连续型空间各个维度的取值范围
     */
    private double[][] spaces;

    public BoxStateSpace(double[][] spaces) {
        Validate.isTrue(checkValid(spaces), "box state space data is invalid!!");
        this.spaces = spaces;
    }

    @Override
    public int getDim() {
        return spaces.length;
    }

    /**
     * 检验数据合法性
     */
    private boolean checkValid(double[][] spaces) {
        if (spaces == null) {
            return false;
        }
        if (spaces.length <= 0) {
            return false;
        }
        for (double[] space : spaces) {
            if (space.length != 2) {
                // 空间的每个维度数据都表示该维度所允许的取值范围；
                // 所以每个维度数据的长度都应为2，分别为最小值和最大值
                return false;
            }
        }
        return true;
    }
}
