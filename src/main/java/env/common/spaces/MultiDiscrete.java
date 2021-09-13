package env.common.spaces;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.lang3.Validate;

import java.util.Collection;
import java.util.List;

/*_
 * 多维离散型空间
 *  e.g. Nintendo Game Controller
 *     - Can be conceptualized as 3 discrete action spaces:
 *
 *         1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
 *         2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
 *         3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
 *
 *     - Can be initialized as
 *
 *         MultiDiscrete([ 5, 2, 2 ])
 *
 * @author Caojunqi
 * @date 2021-09-13 11:09
 */
public class MultiDiscrete implements Space {

    /**
     * 离散型数据的取值，该数组长度表示数据维度，数据值表示对应维度的离散数据取值最大值（不含）。
     */
    private int[] counts;

    public MultiDiscrete(int[] counts) {
        Validate.isTrue(counts != null, "multi-discrete spaces data is invalid!!");
        this.counts = counts;
    }

    @Override
    public boolean contains(Object x) {
        // 校验数据类型合法性
        if (!(x.getClass().isArray()) && !(x instanceof Collection)) {
            return false;
        }
        List<Integer> xList = new ObjectMapper().convertValue(x, new TypeReference<List<Integer>>() {
        });
        if (xList.size() != counts.length) {
            return false;
        }

        // 校验取值范围合法性
        for (int i = 0; i < xList.size(); i++) {
            Integer item = xList.get(i);
            if (item == null) {
                return false;
            }
            if (item < 0 || item >= counts[i]) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int getDim() {
        return this.counts.length;
    }
}
