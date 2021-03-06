package algorithm.ppo2;

import ai.djl.engine.Engine;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.translate.Batchifier;
import algorithm.CommonParameter;
import algorithm.RlAgentCloseable;
import algorithm.ppo.PPOParameter;
import utils.Helper;

import java.util.Arrays;
import java.util.Random;

/**
 * PPO算法 Proximal policy optimization algorithms
 *
 * @author Caojunqi
 * @date 2021-11-27 11:09
 */
public class PPO implements RlAgentCloseable {

    private NDManager agentManager;
    private Random random;
    private Trainer trainer;
    private Batchifier batchifier;

    public PPO(NDManager agentManager, Random random, Trainer trainer) {
        this.agentManager = agentManager;
        this.random = random;
        this.trainer = trainer;
        this.batchifier = Batchifier.STACK;
    }

    @Override
    public NDList chooseAction(RlEnv env, boolean training) {
        try (NDManager manager = agentManager.newSubManager()) {
            NDList[] inputs = buildInputs(manager, env);
            NDArray action;
            if (training) {
                action = trainer.forward(batchifier.batchify(inputs)).get(0).duplicate();
            } else {
                action = trainer.evaluate(batchifier.batchify(inputs)).get(0).duplicate();
            }
            return new NDList(action);
        }
    }

    private NDList[] buildInputs(NDManager manager, RlEnv env) {
        NDList envObservation = env.getObservation();
        envObservation.attach(manager);
        return new NDList[]{envObservation};
    }

    @Override
    public void trainBatch(RlEnv.Step[] batchSteps) {
        try (NDManager subManager = agentManager.newSubManager()) {
            NDArray states = buildBatchPreObservation(batchSteps).singletonOrThrow();
            NDArray actions = buildBatchAction(batchSteps);
            NDArray rewards = buildBatchReward(batchSteps);
            boolean[] masks = buildBatchDone(batchSteps);
            NDArray lastState = batchSteps[batchSteps.length - 1].getPostObservation().singletonOrThrow().expandDims(0);

            NDList output = trainer.forward(new NDList(states));
            NDArray actionsPred = output.get(0);
            NDArray values = output.get(1).duplicate();
            NDArray actionLogProbPred = output.get(2).duplicate();
            NDArray entropy = output.get(3).duplicate();
            NDArray distribution = actionLogProbPred.get(new NDIndex().addAllDim(actionLogProbPred.getShape().dimension() - 1).addPickDim(actions));

            NDList lastValueOutput = trainer.forward(new NDList(lastState));
            float lastValue = lastValueOutput.get(1).duplicate().getFloat(-1);

            NDList estimates = estimateAdvantage(lastValue, values, rewards, masks);
            NDArray expectedReturns = estimates.get(0);
            NDArray advantages = estimates.get(1);

            int optimIterNum = (int) (states.getShape().get(0) + CommonParameter.INNER_BATCH_SIZE - 1) / CommonParameter.INNER_BATCH_SIZE;
            for (int i = 0; i < CommonParameter.INNER_UPDATES; i++) {
                int[] allIndex = subManager.arange((int) states.getShape().get(0)).toIntArray();
                Helper.shuffleArray(allIndex);
                for (int j = 0; j < optimIterNum; j++) {
                    int[] index = Arrays.copyOfRange(allIndex, j * CommonParameter.INNER_BATCH_SIZE, Math.min((j + 1) * CommonParameter.INNER_BATCH_SIZE, (int) states.getShape().get(0)));
                    NDArray statesSubset = getSample(subManager, states, index);
                    NDArray actionsSubset = getSample(subManager, actions, index);
                    NDArray actionLogProbPredSubset = getSample(subManager, actionLogProbPred, index);
                    NDArray entropySubset = getSample(subManager, entropy, index);
                    NDArray expectedReturnsSubset = getSample(subManager, expectedReturns, index);
                    NDArray advantagesSubset = getSample(subManager, advantages, index);
                    NDArray distributionSubset = getSample(subManager, distribution, index);

                    NDList outputBatch = trainer.forward(new NDList(statesSubset));
                    NDArray actionsBatch = outputBatch.get(0);
                    NDArray valuesBatch = outputBatch.get(1);
                    NDArray actionLogProbBatch = outputBatch.get(2);
                    NDArray distributionBatch = actionLogProbBatch.get(new NDIndex().addAllDim(actionLogProbBatch.getShape().dimension() - 1).addPickDim(actionsSubset));

                    NDArray ratios = distributionBatch.sub(distributionSubset).exp();

                    NDArray surr1 = ratios.mul(advantagesSubset);
                    NDArray surr2 = ratios.clip(PPOParameter.RATIO_LOWER_BOUND, PPOParameter.RATIO_UPPER_BOUND).mul(advantagesSubset);
                    NDArray lossActor = surr1.minimum(surr2).mean().neg();

                    NDArray lossCritic = (expectedReturnsSubset.sub(valuesBatch)).square().mean();

                    NDArray lossEntropy = entropySubset.mean().neg();

                    NDArray loss = lossActor.add(lossEntropy.mul(PPOParameter.ENT_LOSS_COEF)).add(lossCritic.mul(PPOParameter.VF_LOSS_COEF));

                    try (GradientCollector collector = Engine.getInstance().newGradientCollector()) {
                        collector.backward(loss);
                        trainer.step();
                    }
                }
            }

            trainer.notifyListeners(listener -> listener.onTrainingBatch(trainer, null));
        }
    }

    @Override
    public void close() {
        this.agentManager.close();
        this.trainer.close();
    }

    private NDList buildBatchPreObservation(RlEnv.Step[] batchSteps) {
        NDList[] result = new NDList[batchSteps.length];
        for (int i = 0; i < batchSteps.length; i++) {
            result[i] = batchSteps[i].getPreObservation();
        }
        return batchifier.batchify(result);
    }

    private NDArray buildBatchAction(RlEnv.Step[] batchSteps) {
        NDList[] result = new NDList[batchSteps.length];
        for (int i = 0; i < batchSteps.length; i++) {
            result[i] = batchSteps[i].getAction();
        }
        return batchifier.batchify(result).singletonOrThrow();
    }

    public NDArray buildBatchReward(RlEnv.Step[] batchSteps) {
        NDList[] result = new NDList[batchSteps.length];
        for (int i = 0; i < batchSteps.length; i++) {
            result[i] = new NDList(batchSteps[i].getReward().expandDims(0));
        }
        return batchifier.batchify(result).singletonOrThrow();
    }

    public boolean[] buildBatchDone(RlEnv.Step[] batchSteps) {
        boolean[] resultData = new boolean[batchSteps.length];
        for (int i = 0; i < batchSteps.length; i++) {
            resultData[i] = batchSteps[i].isDone();
        }
        return resultData;
    }

    private NDList estimateAdvantage(float lastValue, NDArray values, NDArray rewards, boolean[] masks) {
        NDManager manager = rewards.getManager();
        NDArray deltas = manager.create(rewards.getShape());
        NDArray advantages = manager.create(rewards.getShape());

        float prevValue = lastValue;
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
