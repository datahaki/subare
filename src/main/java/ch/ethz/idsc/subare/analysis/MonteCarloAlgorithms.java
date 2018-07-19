// code by fluric
package ch.ethz.idsc.subare.analysis;

import java.util.List;

import ch.ethz.idsc.subare.core.DiscreteQsaSupplier;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.td.DoubleSarsa;
import ch.ethz.idsc.subare.core.td.ExpectedSarsa;
import ch.ethz.idsc.subare.core.td.ExpectedTrueOnlineSarsa;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.OriginalTrueOnlineSarsa;
import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.td.QLearningTrueOnlineSarsa;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.SarsaType;
import ch.ethz.idsc.subare.core.td.TrueOnlineSarsa;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DefaultLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.subare.util.Stopwatch;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.red.Mean;

public enum MonteCarloAlgorithms {
  ORIGINAL_SARSA() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      DiscreteQsa qsaSarsa = DiscreteQsa.build(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.05));
      final Sarsa sarsa = new OriginalSarsa(monteCarloInterface, qsaSarsa, learningRate);
      sarsa.setExplore(RealScalar.of(0.1));
      return sarsa;
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof OriginalSarsa);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, algorithm.qsa(), RealScalar.of(0.1));
      ExploringStarts.batch(monteCarloInterface, policy, 1, (OriginalSarsa) algorithm);
    }
  }, //
  EXPECTED_SARSA() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      DiscreteQsa qsaSarsa = DiscreteQsa.build(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.05));
      final Sarsa sarsa = new ExpectedSarsa(monteCarloInterface, qsaSarsa, learningRate);
      sarsa.setExplore(RealScalar.of(0.1));
      return sarsa;
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof ExpectedSarsa);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, algorithm.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(monteCarloInterface, policy, 1, (ExpectedSarsa) algorithm);
    }
  }, //
  QLEARNING_SARSA() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      DiscreteQsa qsaSarsa = DiscreteQsa.build(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.05));
      final Sarsa sarsa = new QLearning(monteCarloInterface, qsaSarsa, learningRate);
      sarsa.setExplore(RealScalar.of(0.1));
      return sarsa;
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof QLearning);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, algorithm.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(monteCarloInterface, policy, 1, (QLearning) algorithm);
    }
  }, //
  DOUBLE_QLEARNING_SARSA() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface);
      DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface);
      DoubleSarsa sarsa = new DoubleSarsa(SarsaType.QLEARNING, monteCarloInterface, //
          qsa1, qsa2, //
          DefaultLearningRate.of(1, .51), //
          DefaultLearningRate.of(1, .51));
      sarsa.setExplore(RealScalar.of(0.1));
      return sarsa;
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof DoubleSarsa);
      Policy policy = ((DoubleSarsa) algorithm).getEGreedy();
      ExploringStarts.batch(monteCarloInterface, policy, 1, (DoubleSarsa) algorithm);
    }
  }, //
  MONTE_CARLO() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      return new MonteCarloExploringStarts(monteCarloInterface);
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof MonteCarloExploringStarts);
      Policy policyMC = EGreedyPolicy.bestEquiprobable(monteCarloInterface, algorithm.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(monteCarloInterface, policyMC, (MonteCarloExploringStarts) algorithm);
    }
  }, //
  ORIGINAL_TRUE_ONLINE_SARSA() {// choose ConstantLearningRate.of(RealScalar.of(0.05)) or ConstantLearningRate.of(RealScalar.of(0.1))
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.05));
      TrueOnlineSarsa trueOnlineSarsa = OriginalTrueOnlineSarsa.of(monteCarloInterface, RealScalar.of(0.3), learningRate, featureMapper);
      trueOnlineSarsa.setExplore(RealScalar.of(0.1));
      return trueOnlineSarsa;
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof TrueOnlineSarsa);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, algorithm.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(monteCarloInterface, policy, (TrueOnlineSarsa) algorithm);
    }
  }, //
  EXPECTED_TRUE_ONLINE_SARSA() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.05));
      TrueOnlineSarsa trueOnlineSarsa = ExpectedTrueOnlineSarsa.of(monteCarloInterface, RealScalar.of(0.3), learningRate, featureMapper);
      trueOnlineSarsa.setExplore(RealScalar.of(0.1));
      return trueOnlineSarsa;
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof TrueOnlineSarsa);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, algorithm.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(monteCarloInterface, policy, (TrueOnlineSarsa) algorithm);
    }
  }, //
  QLEARNING_TRUE_ONLINE_SARSA() {
    @Override
    public DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface) {
      FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RealScalar.of(0.05));
      TrueOnlineSarsa trueOnlineSarsa = QLearningTrueOnlineSarsa.of(monteCarloInterface, RealScalar.of(0.3), learningRate, featureMapper);
      trueOnlineSarsa.setExplore(RealScalar.of(0.1));
      return trueOnlineSarsa;
    }

    @Override
    public void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface) {
      GlobalAssert.that(algorithm instanceof TrueOnlineSarsa);
      Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, algorithm.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(monteCarloInterface, policy, (TrueOnlineSarsa) algorithm);
    }
  }, //
  ;
  public abstract DiscreteQsaSupplier getAlgorithm(MonteCarloInterface monteCarloInterface);

  public abstract void executeBatch(DiscreteQsaSupplier algorithm, MonteCarloInterface monteCarloInterface);
  // public abstract Tensor analyse(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa, List<MonteCarloErrorAnalysis> errorAnalysis);

  public Tensor analyseNTimes(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa, List<DiscreteModelErrorAnalysis> errorAnalysis,
      int nTimes) {
    Tensor nSamples = Tensors.empty();
    Stopwatch stopwatch = Stopwatch.started();
    Stopwatch subWatch = Stopwatch.started();
    for (int i = 0; i < nTimes; ++i) {
      nSamples.append(analyseAlgorithm(monteCarloInterface, batches, optimalQsa, errorAnalysis));
      if (subWatch.display_seconds() > 10.0) {
        System.out.println(this.name() + " has finished trial " + i);
        subWatch = Stopwatch.started();
      }
    }
    System.out.println("Time for executing " + this.name() + " " + nTimes + " times with " + batches + " batches: " + stopwatch.display_seconds() + "s");
    return Mean.of(nSamples);
  }

  private Tensor analyseAlgorithm(MonteCarloInterface monteCarloInterface, int batches, DiscreteQsa optimalQsa,
      List<DiscreteModelErrorAnalysis> errorAnalysisList) {
    Tensor samples = Tensors.empty();
    DiscreteQsaSupplier algorithm = getAlgorithm(monteCarloInterface);
    for (int index = 0; index < batches; ++index) {
      Tensor sample = Tensors.vector(index);
      // System.out.println("starting batch " + (index + 1) + " of " + batches);
      executeBatch(algorithm, monteCarloInterface);
      for (DiscreteModelErrorAnalysis errorAnalysis : errorAnalysisList) {
        sample.append(errorAnalysis.getError(monteCarloInterface, optimalQsa, algorithm.qsa()));
      }
      samples.append(sample);
    }
    return samples;
  }
}