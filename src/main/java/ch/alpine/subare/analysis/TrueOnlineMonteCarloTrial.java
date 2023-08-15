// code by fluric
package ch.alpine.subare.analysis;

import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.core.QsaInterface;
import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.StepRecord;
import ch.alpine.subare.core.td.SarsaType;
import ch.alpine.subare.core.td.TrueOnlineSarsa;
import ch.alpine.subare.core.util.ConstantLearningRate;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteStateActionCounter;
import ch.alpine.subare.core.util.ExactFeatureMapper;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.subare.core.util.FeatureMapper;
import ch.alpine.subare.core.util.FeatureWeight;
import ch.alpine.subare.core.util.LearningRate;
import ch.alpine.subare.core.util.PolicyBase;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;

/* package */ class TrueOnlineMonteCarloTrial implements MonteCarloTrial {
  // TODO SUBARE make configurable
  private static final Scalar ALPHA = RealScalar.of(0.05);
  private static final Scalar LAMBDA = RealScalar.of(0.3);

  public static MonteCarloTrial of(MonteCarloInterface monteCarloInterface, SarsaType sarsaType) {
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    QsaInterface qsa = DiscreteQsa.build(monteCarloInterface);
    StateActionCounter sac = new DiscreteStateActionCounter();
    return new TrueOnlineMonteCarloTrial(monteCarloInterface, sarsaType, featureMapper, ConstantLearningRate.of(ALPHA), new FeatureWeight(featureMapper), sac,
        PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac));
  }

  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final TrueOnlineSarsa trueOnlineSarsa;
  private final PolicyBase policy;

  private TrueOnlineMonteCarloTrial(MonteCarloInterface monteCarloInterface, SarsaType sarsaType, //
      FeatureMapper featureMapper, LearningRate learningRate, FeatureWeight w, StateActionCounter sac, PolicyBase policy) {
    this.monteCarloInterface = monteCarloInterface;
    this.policy = policy;
    trueOnlineSarsa = sarsaType.trueOnline(monteCarloInterface, LAMBDA, featureMapper, learningRate, w, sac, policy);
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    policy.setQsa(trueOnlineSarsa.qsaInterface());
    ExploringStarts.batch(monteCarloInterface, policy, trueOnlineSarsa);
  }

  @Override // from MonteCarloTrial
  public DiscreteQsa qsa() {
    return trueOnlineSarsa.qsa();
  }

  @Override // from MonteCarloTrial
  public void digest(StepRecord stepInterface) {
    trueOnlineSarsa.digest(stepInterface);
  }

  @Override // from MonteCarloTrial
  public QsaInterface qsaInterface() {
    return trueOnlineSarsa.qsaInterface();
  }
}
