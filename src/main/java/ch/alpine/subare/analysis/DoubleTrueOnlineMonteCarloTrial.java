// code by fluric
package ch.alpine.subare.analysis;

import ch.alpine.subare.api.FeatureMapper;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.api.QsaInterface;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StateActionCounterSupplier;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.td.DoubleTrueOnlineSarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.ExactFeatureMapper;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.FeatureWeight;
import ch.alpine.subare.util.PolicyBase;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;

/* package */ class DoubleTrueOnlineMonteCarloTrial implements MonteCarloTrial, StateActionCounterSupplier {
  private static final Scalar ALPHA = RealScalar.of(0.05);
  private static final Scalar LAMBDA = RealScalar.of(0.3);

  public static MonteCarloTrial create(MonteCarloInterface monteCarloInterface, SarsaType sarsaType) {
    DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface);
    DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface);
    StateActionCounter sac1 = new DiscreteStateActionCounter();
    StateActionCounter sac2 = new DiscreteStateActionCounter();
    PolicyBase policy1 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa1, sac1);
    PolicyBase policy2 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa2, sac2);
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    return new DoubleTrueOnlineMonteCarloTrial(monteCarloInterface, sarsaType, featureMapper, //
        ConstantLearningRate.of(ALPHA), sac1, sac2, //
        new FeatureWeight(featureMapper), new FeatureWeight(featureMapper), //
        policy1, policy2);
  }

  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final DoubleTrueOnlineSarsa doubleTrueOnlineSarsa;

  // has convergence problems, don't use it yet!
  private DoubleTrueOnlineMonteCarloTrial( //
      MonteCarloInterface monteCarloInterface, SarsaType sarsaType, //
      FeatureMapper featureMapper, //
      LearningRate learningRate, //
      StateActionCounter sac1, StateActionCounter sac2, //
      FeatureWeight w1, FeatureWeight w2, //
      PolicyBase policy1, PolicyBase policy2) {
    this.monteCarloInterface = monteCarloInterface;
    doubleTrueOnlineSarsa = sarsaType.doubleTrueOnline(monteCarloInterface, LAMBDA, featureMapper, learningRate, w1, w2, sac1, sac2, policy1, policy2);
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    ExploringStarts.batch(monteCarloInterface, doubleTrueOnlineSarsa.getPolicy(), doubleTrueOnlineSarsa);
  }

  @Override // from MonteCarloTrial
  public DiscreteQsa qsa() {
    return doubleTrueOnlineSarsa.qsa();
  }

  @Override // from MonteCarloTrial
  public void digest(StepRecord stepInterface) {
    doubleTrueOnlineSarsa.digest(stepInterface);
  }

  @Override // from MonteCarloTrial
  public QsaInterface qsaInterface() {
    return doubleTrueOnlineSarsa.qsaInterface();
  }

  @Override // from StateActionCounterSupplier
  public StateActionCounter sac() {
    return doubleTrueOnlineSarsa.sac();
  }
}
