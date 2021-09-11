package env.common.spaces;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.lang3.Validate;

import java.util.Collection;
import java.util.List;

/**
 * 连续型空间
 * A (possibly unbounded) box in R^n. Specifically, a Box represents the
 * Cartesian product of n closed intervals. Each interval has the form of one
 * of [a, b], (-oo, b], [a, oo), or (-oo, oo).
 *
 * @author Caojunqi
 * @date 2021-09-11 15:40
 */
public class Box implements Space {
    /**
     * 连续型空间各个维度的取值范围
     */
    private double[][] spaces;

    public Box(double[][] spaces) {
        Validate.isTrue(checkValid(spaces), "spaces data is invalid!!");
        this.spaces = spaces;
    }

    @Override
    public boolean contains(Object x) {
        // 校验数据类型合法性
        if (!(x.getClass().isArray()) && !(x instanceof Collection)) {
            return false;
        }
        List<Double> xList = new ObjectMapper().convertValue(x, new TypeReference<List<Double>>() {
        });
        if (xList.size() != spaces.length) {
            return false;
        }

        // 校验取值范围合法性
        for (int i = 0; i < xList.size(); i++) {
            Double item = xList.get(i);
            if (item == null) {
                return false;
            }
            if (item < spaces[i][0] || item > spaces[i][1]) {
                return false;
            }
        }
        return true;
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
