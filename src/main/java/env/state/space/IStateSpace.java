package env.state.space;

import env.state.core.IState;

/**
 * 状态空间接口
 *
 * @author Caojunqi
 * @date 2021-09-13 12:06
 */
public interface IStateSpace<S extends IState> {

    /**
     * 状态空间维度
     */
    int getDim();

}
