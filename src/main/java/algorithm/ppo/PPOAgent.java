package algorithm.ppo;

import ai.djl.engine.Engine;
import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.translate.Batchifier;
import algorithm.CommonParameter;
import utils.ActionSampler;
import utils.Helper;

import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author Caojunqi
 * @date 2021-11-29 20:46
 */
public class PPOAgent implements RlAgent {

    private Random random;
    private Trainer policyTrainer;
    private Trainer valueTrainer;
    private float rewardDiscount;
    private Batchifier batchifier;

    public PPOAgent(Random random, Trainer policyTrainer, Trainer valueTrainer, float rewardDiscount) {
        this.random = random;
        this.policyTrainer = policyTrainer;
        this.valueTrainer = valueTrainer;
        this.rewardDiscount = rewardDiscount;
        this.batchifier = Batchifier.STACK;
    }

    @Override
    public NDList chooseAction(RlEnv env, boolean training) {
        NDList[] inputs = buildInputs(env.getObservation());
        NDArray actionScores =
                policyTrainer.evaluate(batchifier.batchify(inputs)).singletonOrThrow().squeeze(-1);
        int action;
        if (training) {
            action = ActionSampler.sampleMultinomial(actionScores, random);
        } else {
            action = Math.toIntExact(actionScores.argMax().getLong());
        }
        ActionSpace actionSpace = env.getActionSpace();
        return actionSpace.get(action);
    }

    @Override
    public void trainBatch(RlEnv.Step[] batchSteps) {
        TrainingListener.BatchData batchData =
                new TrainingListener.BatchData(null, new ConcurrentHashMap<>(), new ConcurrentHashMap<>());
        NDList[] preObservations = buildBatchPreObservation(batchSteps);
        NDList[] actions = buildBatchAction(batchSteps);
        NDList[] rewards = buildBatchReward(batchSteps);
        boolean[] dones = buildBatchDone(batchSteps);

        NDList policyOutput = policyTrainer.evaluate(batchifier.batchify(preObservations));
        NDArray distribution = Helper.gather(policyOutput.singletonOrThrow().duplicate(), batchifier.batchify(actions).singletonOrThrow().toIntArray());

        NDList valueOutput = valueTrainer.evaluate(batchifier.batchify(preObservations));
        NDArray values = valueOutput.singletonOrThrow().duplicate();
        NDList estimates = estimateAdvantage(values.duplicate(), batchifier.batchify(rewards).singletonOrThrow(), dones);
        NDArray expectedReturns = estimates.get(0);
        NDArray advantages = estimates.get(1);

        // update critic
        NDList valueOutputUpdated = valueTrainer.forward(batchifier.batchify(preObservations));
        NDArray valuesUpdated = valueOutputUpdated.singletonOrThrow();
        NDArray lossCritic = (expectedReturns.sub(valuesUpdated)).square().mean();
        try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
            collector.backward(lossCritic);
        }

        // update policy
        NDList policyOutputUpdated = policyTrainer.forward(batchifier.batchify(preObservations));
        NDArray distributionUpdated = Helper.gather(policyOutputUpdated.singletonOrThrow(), batchifier.batchify(actions).singletonOrThrow().toIntArray());
        NDArray ratios = distributionUpdated.div(distribution);

        NDArray surr1 = ratios.mul(advantages);
        NDArray surr2 = ratios.clip(PPOParameter.RATIO_LOWER_BOUND, PPOParameter.RATIO_UPPER_BOUND).mul(advantages);
        NDArray lossActor = surr1.minimum(surr2).mean().neg();

        try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
            collector.backward(lossActor);
        }


        policyTrainer.notifyListeners(listener -> listener.onTrainingBatch(policyTrainer, batchData));
        valueTrainer.notifyListeners(listener -> listener.onTrainingBatch(valueTrainer, batchData));
    }

    private NDList[] buildInputs(NDList observation) {
        return new NDList[]{observation};
    }

    public NDList[] buildBatchPreObservation(RlEnv.Step[] batchSteps) {
        NDList[] result = new NDList[batchSteps.length];
        for (int i = 0; i < batchSteps.length; i++) {
            result[i] = batchSteps[i].getPreObservation();
        }
        return result;
    }

    public NDList[] buildBatchAction(RlEnv.Step[] batchSteps) {
        NDList[] result = new NDList[batchSteps.length];
        for (int i = 0; i < batchSteps.length; i++) {
            result[i] = batchSteps[i].getAction();
        }
        return result;
    }

    public NDList[] buildBatchPostObservation(RlEnv.Step[] batchSteps) {
        NDList[] result = new NDList[batchSteps.length];
        for (int i = 0; i < batchSteps.length; i++) {
            result[i] = batchSteps[i].getPostObservation();
        }
        return result;
    }

    public NDList[] buildBatchReward(RlEnv.Step[] batchSteps) {
        NDList[] result = new NDList[batchSteps.length];
        for (int i = 0; i < batchSteps.length; i++) {
            result[i] = new NDList(batchSteps[i].getReward());
        }
        return result;
    }

    public boolean[] buildBatchDone(RlEnv.Step[] batchSteps) {
        boolean[] resultData = new boolean[batchSteps.length];
        for (int i = 0; i < batchSteps.length; i++) {
            resultData[i] = batchSteps[i].isDone();
        }
        return resultData;
    }

    private NDList estimateAdvantage(NDArray values, NDArray rewards, boolean[] masks) {
        NDManager manager = rewards.getManager();
        NDArray deltas = manager.create(rewards.getShape());
        NDArray advantages = manager.create(rewards.getShape());

        float prevValue = 0;
        float prevAdvantage = 0;
        for (int i = (int) rewards.getShape().get(0) - 1; i >= 0; i--) {
            NDIndex index = new NDIndex(i);
            int mask = masks[i] ? 0 : 1;
            deltas.set(index, rewards.get(i).add(CommonParameter.GAMMA * prevValue * mask).sub(values.get(i)));
            advantages.set(index, deltas.get(i).add(CommonParameter.GAMMA * CommonParameter.GAE_LAMBDA * prevAdvantage * mask));

            prevValue = values.getFloat(i);
            prevAdvantage = advantages.getFloat(i);
        }

        NDArray expected_returns = values.add(advantages);
        NDArray advantagesMean = advantages.mean();
        NDArray advantagesStd = advantages.sub(advantagesMean).pow(2).sum().div(advantages.size() - 1).sqrt();
        advantages = advantages.sub(advantagesMean).div(advantagesStd);

        return new NDList(expected_returns, advantages);
    }

    private NDArray getSample(NDManager subManager, NDArray array, int[] index) {
        Shape shape = Shape.update(array.getShape(), 0, index.length);
        NDArray sample = subManager.zeros(shape, array.getDataType());
        for (int i = 0; i < index.length; i++) {
            sample.set(new NDIndex(i), array.get(index[i]));
        }
        return sample;
    }


}
