package demo.cartpole;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;
import algorithm.BaseModelBlock;
import algorithm.CommonParameter;
import algorithm.ppo.PPOAgent;
import algorithm.ppo.PPOParameter;
import algorithm.ppo.block.CriticValueModelBlock;
import algorithm.ppo.block.DiscretePolicyModelBlock;
import common.Arguments;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Random;

/**
 * @author Caojunqi
 * @date 2021-11-29 20:30
 */
public class TrainCartPole {

    private static final Logger logger = LoggerFactory.getLogger(TrainCartPole.class);

    private TrainCartPole() {
    }

    public static void main(String[] args) throws IOException {
        TrainCartPole.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return null;
        }

        Engine.getInstance().setRandomSeed(0);
        int epoch = 500;
        int batchSize = 64;
        int replayBufferSize = 2048;
        int gamesPerEpoch = 128;
        // Validation is deterministic, thus one game is enough
        int validationGamesPerEpoch = 1;
        float rewardDiscount = 0.9f;

        if (arguments.getLimit() != Long.MAX_VALUE) {
            gamesPerEpoch = Math.toIntExact(arguments.getLimit());
        }

        Random random = new Random(0);
        NDManager mainManager = NDManager.newBaseManager();
        CartPole env = new CartPole(mainManager, random, batchSize, replayBufferSize);
        int stateSpaceDim = (int) env.getObservation().singletonOrThrow().getShape().get(0);
        int actionSpaceDim = env.getActionSpace().size();
        Model policyModel = Model.newInstance("discrete_policy_model");
        BaseModelBlock policyNet = new DiscretePolicyModelBlock(actionSpaceDim, PPOParameter.POLICY_MODEL_HIDDEN_SIZE);
        policyModel.setBlock(policyNet);
        Model valueModel = Model.newInstance("critic_value_model");
        BaseModelBlock valueNet = new CriticValueModelBlock(PPOParameter.CRITIC_MODEL_HIDDEN_SIZE);
        valueModel.setBlock(valueNet);

        DefaultTrainingConfig policyConfig = setupTrainingConfig();
        DefaultTrainingConfig valueConfig = setupTrainingConfig();
        Trainer policyTrainer = policyModel.newTrainer(policyConfig);
        Trainer valueTrainer = valueModel.newTrainer(valueConfig);

        policyTrainer.initialize(new Shape(batchSize, 4));
        valueTrainer.initialize(new Shape(batchSize, 4));
        policyTrainer.notifyListeners(listener -> listener.onTrainingBegin(policyTrainer));
        valueTrainer.notifyListeners(listener -> listener.onTrainingBegin(valueTrainer));

        PPOAgent agent = new PPOAgent(random, policyTrainer, valueTrainer, rewardDiscount);
        for (int i = 0; i < epoch; i++) {

            int episode = 0;
            int size = 0;
            while (size < replayBufferSize) {
                episode++;
                float result = env.runEnvironment(agent, true);
                size += (int) result;
                System.out.println("[" + episode + "]train:" + result);
            }

            for (int j = 0; j < gamesPerEpoch; j++) {
                RlEnv.Step[] batchSteps = env.getBatch();
                agent.trainBatch(batchSteps);
                policyTrainer.step();
                valueTrainer.step();
//                System.out.println("train[" + i + "-" + j + "]:" + result);
            }

            for (int j = 0; j < validationGamesPerEpoch; j++) {
                float result = env.runEnvironment(agent, false);
                System.out.println("test:" + result);
            }
        }

        policyTrainer.notifyListeners(listener -> listener.onTrainingEnd(policyTrainer));
        valueTrainer.notifyListeners(listener -> listener.onTrainingEnd(valueTrainer));

        return policyTrainer.getTrainingResult();
    }

    public static DefaultTrainingConfig setupTrainingConfig() {
        return new DefaultTrainingConfig(Loss.l2Loss())
                .addTrainingListeners(TrainingListener.Defaults.basic())
                .optOptimizer(
                        Adam.builder().optLearningRateTracker(Tracker.fixed(CommonParameter.LEARNING_RATE)).build());
    }

}
