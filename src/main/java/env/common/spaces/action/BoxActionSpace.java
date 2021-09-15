package env.common.spaces.action;

import env.common.action.impl.BoxAction;
import org.apache.commons.lang3.Validate;

/**
 * 连续型动作空间
 *
 * @author Caojunqi
 * @date 2021-09-13 11:57
 */
public class BoxActionSpace implements ActionSpace<BoxAction> {

    /**
     * 连续型空间各个维度的取值范围
     */
    private double[][] spaces;

    public BoxActionSpace(double[][] spaces) {
        Validate.isTrue(checkValid(spaces), "box action space data is invalid!!");
        this.spaces = spaces;
    }

    @Override
    public boolean canStep(BoxAction action) {
        double[] actionData = action.getActionData();
        for (int i = 0; i < actionData.length; i++) {
            if (actionData[i] < spaces[i][0] || actionData[i] > spaces[i][1]) {
                return false;
            }
        }
        return true;
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
