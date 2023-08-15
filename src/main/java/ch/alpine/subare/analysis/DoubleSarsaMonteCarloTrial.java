// code by fluric
package ch.alpine.subare.analysis;

import java.util.ArrayDeque;
import java.util.Deque;

import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.QsaInterface;
import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.StepRecord;
import ch.alpine.subare.core.td.DoubleSarsa;
import ch.alpine.subare.core.td.SarsaType;
import ch.alpine.subare.core.util.ConstantLearningRate;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteStateActionCounter;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.subare.core.util.LearningRate;
import ch.alpine.subare.core.util.PolicyBase;
import ch.alpine.subare.core.util.PolicyExt;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.tensor.RealScalar;

/* package */ class DoubleSarsaMonteCarloTrial implements MonteCarloTrial {
  public static MonteCarloTrial of(MonteCarloInterface monteCarloInterface, SarsaType sarsaType) {
    DiscreteQsa qsa1 = DiscreteQsa.build(monteCarloInterface);
    DiscreteQsa qsa2 = DiscreteQsa.build(monteCarloInterface);
    StateActionCounter sac1 = new DiscreteStateActionCounter();
    StateActionCounter sac2 = new DiscreteStateActionCounter();
    PolicyBase policy1 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa1, sac1);
    PolicyBase policy2 = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa2, sac2);
    return new DoubleSarsaMonteCarloTrial(monteCarloInterface, sarsaType, //
        ConstantLearningRate.of(RealScalar.of(0.05)), qsa1, qsa2, sac1, sac2, policy1, policy2);
  }

  private final static int DIGEST_DEPTH = 1;
  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final DoubleSarsa doubleSarsa;
  private final Deque<StepRecord> deque = new ArrayDeque<>();

  public DoubleSarsaMonteCarloTrial( //
      MonteCarloInterface monteCarloInterface, SarsaType sarsaType, //
      LearningRate learningRate, //
      DiscreteQsa qsa1, DiscreteQsa qsa2, //
      StateActionCounter sac1, StateActionCounter sac2, //
      PolicyExt policy1, PolicyExt policy2) {
    this.monteCarloInterface = monteCarloInterface;
    doubleSarsa = sarsaType.doubleSarsa(monteCarloInterface, //
        learningRate, qsa1, qsa2, sac1, sac2, policy1, policy2);
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    Policy policy = doubleSarsa.getPolicy();
    ExploringStarts.batch(monteCarloInterface, policy, DIGEST_DEPTH, doubleSarsa);
  }

  @Override // from MonteCarloTrial
  public DiscreteQsa qsa() {
    return doubleSarsa.qsa();
  }

  @Override // from MonteCarloTrial
  public void digest(StepRecord stepInterface) {
    deque.add(stepInterface);
    if (!monteCarloInterface.isTerminal(stepInterface.nextState())) {
      if (deque.size() == DIGEST_DEPTH) { // never true, if nstep == 0
        doubleSarsa.digest(deque);
        deque.poll();
      }
    } else {
      while (!deque.isEmpty()) {
        doubleSarsa.digest(deque);
        deque.poll();
      }
    }
  }

  @Override // from MonteCarloTrial
  public QsaInterface qsaInterface() {
    return qsa();
  }
}
