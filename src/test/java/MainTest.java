import utils.datatype.Transition;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * 测试类
 *
 * @author Caojunqi
 * @date 2021-09-10 10:06
 */
public class MainTest {

    public static void main(String[] args) {
        Transition[] memory = new Transition[3];
        memory[0] = new Transition(new float[]{1.1f, 1.1f}, 1, true, new float[]{2.1f, 2.1f}, 3.1f);
        memory[1] = new Transition(new float[]{1.2f, 1.2f}, 2, true, new float[]{2.2f, 2.2f}, 3.2f);
        memory[2] = new Transition(new float[]{1.3f, 1.3f}, 3, true, new float[]{2.3f, 2.3f}, 3.3f);
        List<Transition> tmpList = new ArrayList<>(Arrays.asList(memory));
        Collections.shuffle(tmpList);
        tmpList.get(0).setAction(10086);
        System.out.println(memory.length);
    }
}
