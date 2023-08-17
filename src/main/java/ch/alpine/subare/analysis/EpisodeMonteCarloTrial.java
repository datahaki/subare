// code by fluric
package ch.alpine.subare.analysis;

import ch.alpine.subare.core.api.MonteCarloInterface;
import ch.alpine.subare.core.api.QsaInterface;
import ch.alpine.subare.core.api.StepRecord;
import ch.alpine.subare.core.mc.MonteCarloExploringStarts;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.subare.core.util.PolicyBase;

/* package */ class EpisodeMonteCarloTrial implements MonteCarloTrial {
  private final MonteCarloInterface monteCarloInterface;
  private final MonteCarloExploringStarts mces;
  private final PolicyBase policy;

  public EpisodeMonteCarloTrial(MonteCarloInterface monteCarloInterface, PolicyBase policy) {
    this.monteCarloInterface = monteCarloInterface;
    this.mces = new MonteCarloExploringStarts(monteCarloInterface);
    this.policy = policy;
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    ExploringStarts.batch(monteCarloInterface, policy, mces);
  }

  @Override // from MonteCarloTrial
  public DiscreteQsa qsa() {
    return mces.qsa();
  }

  @Override // from MonteCarloTrial
  public void digest(StepRecord stepInterface) {
    throw new UnsupportedOperationException();
  }

  @Override // from MonteCarloTrial
  public QsaInterface qsaInterface() {
    return qsa();
  }
}
