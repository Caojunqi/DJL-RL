package algorithm.ppo2;

import ai.djl.modality.rl.LruReplayBuffer;
import ai.djl.modality.rl.ReplayBuffer;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.util.RandomUtils;

/**
 * @author Caojunqi
 * @date 2021-12-01 22:09
 */
public class FixedBuffer implements ReplayBuffer {

    private int batchSize;

    private RlEnv.Step[] steps;
    private int firstStepIndex;
    private int stepsActualSize;

    /**
     * Constructs a {@link LruReplayBuffer}.
     *
     * @param batchSize the number of steps to train on per batch
     * @param bufferSize the number of steps to hold in the buffer
     */
    public FixedBuffer(int batchSize, int bufferSize) {
        this.batchSize = batchSize;
        steps = new RlEnv.Step[bufferSize];
        firstStepIndex = 0;
        stepsActualSize = 0;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("PMD.AvoidArrayLoops")
    public RlEnv.Step[] getBatch() {
        return steps;
    }

    /** {@inheritDoc} */
    @Override
    public void addStep(RlEnv.Step step) {
        if (stepsActualSize == steps.length) {
            int stepToReplace = Math.floorMod(firstStepIndex - 1, steps.length);
            steps[stepToReplace].close();
            steps[stepToReplace] = step;
            firstStepIndex = Math.floorMod(firstStepIndex + 1, steps.length);
        } else {
            steps[stepsActualSize] = step;
            stepsActualSize++;
        }
    }
}
