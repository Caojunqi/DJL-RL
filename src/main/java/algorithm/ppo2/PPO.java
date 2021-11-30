package algorithm.ppo2;

import java.util.Map;

/**
 * PPO算法 Proximal policy optimization algorithms
 *
 * @author Caojunqi
 * @date 2021-11-27 11:09
 */
public class PPO {
    // BaseAlgorithm
    private int verbose;
    private int numTimeSteps;
    private int totalTimeSteps;
    private float learningRate;
    private String tensorBoardLog;
    private int episodeNum;
    private boolean useSde;
    private int sdeSampleFreq;
    private int currentProgressRemaining = 1;
    private int nUpdates;
    // OnPolicyAlgorithm
    private int nSteps;
    private float gamma;
    private float gaeLambda;
    private float entCoef;
    private float vfCoef;
    private float maxGradNorm;

    public PPO(String policy,
               String env,
               float learningRate,
               int nSteps,
               int batchSize,
               int nEpochs,
               float gamma,
               float gaeLambda,
               float clipRange,
               float clipRangeVf,
               float entCoef,
               float vfCoef,
               float maxGradNorm,
               boolean useSde,
               int sdeSampleFreq,
               float targetKl,
               String tensorBoardLog,
               boolean createEvalEnv,
               Map<String, Object> policyKwargs,
               int verbose,
               int seed,
               String device,
               boolean initSetupModel) {
        this.verbose = verbose;
        this.learningRate = learningRate;
        this.tensorBoardLog = tensorBoardLog;
        this.useSde = useSde;
        this.sdeSampleFreq = sdeSampleFreq;

        this.nSteps = nSteps;
        this.gamma = gamma;
        this.gaeLambda = gaeLambda;
        this.entCoef = entCoef;
        this.vfCoef = vfCoef;
        this.maxGradNorm = maxGradNorm;
        if (initSetupModel) {
            setupModel();
        }
    }

    private void setupModel() {

    }
}
