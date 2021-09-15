package env.common.action;

/**
 * 动作数据接口
 *
 * @author Caojunqi
 * @date 2021-09-13 11:24
 */
public interface Action {

    /**
     * 返回对应的动作数据收集器类型
     */
    Class<?> getCollectorClz();

}
