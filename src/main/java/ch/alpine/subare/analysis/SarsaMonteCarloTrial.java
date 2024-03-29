// code by fluric
package ch.alpine.subare.analysis;

import java.util.ArrayDeque;
import java.util.Deque;

import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.QsaInterface;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.PolicyBase;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.sca.Sign;

/** class is used externally */
public class SarsaMonteCarloTrial implements MonteCarloTrial {
  public static MonteCarloTrial of( //
      MonteCarloInterface monteCarloInterface, SarsaType sarsaType, LearningRate learningRate, DiscreteQsa qsa, //
      StateActionCounter sac, PolicyBase policy, int digestDepth) {
    return new SarsaMonteCarloTrial(monteCarloInterface, sarsaType, learningRate, qsa, sac, policy, digestDepth);
  }

  // ---
  private final MonteCarloInterface monteCarloInterface;
  private final Sarsa sarsa;
  private final Deque<StepRecord> deque = new ArrayDeque<>();
  private final PolicyBase policy;
  private final int digestDepth; // 0 is equal to the MonteCarlo approach

  private SarsaMonteCarloTrial( //
      MonteCarloInterface monteCarloInterface, SarsaType sarsaType, LearningRate learningRate, DiscreteQsa qsa, //
      StateActionCounter sac, PolicyBase policy, int digestDepth) {
    this.monteCarloInterface = monteCarloInterface;
    sarsa = sarsaType.sarsa(monteCarloInterface, learningRate, qsa, sac, policy);
    this.policy = policy;
    this.digestDepth = digestDepth;
    Sign.requirePositiveOrZero(RealScalar.of(digestDepth));
  }

  @Override // from MonteCarloTrial
  public void executeBatch() {
    ExploringStarts.batch(monteCarloInterface, policy, digestDepth, sarsa);
  }

  @Override // from MonteCarloTrial
  public DiscreteQsa qsa() {
    return sarsa.qsa();
  }

  @Override // from MonteCarloTrial
  public void digest(StepRecord stepRecord) {
    deque.add(stepRecord);
    if (!monteCarloInterface.isTerminal(stepRecord.nextState())) {
      if (deque.size() == digestDepth) { // never true, if nstep == 0
        sarsa.digest(deque);
        deque.poll();
      }
    } else {
      while (!deque.isEmpty()) {
        sarsa.digest(deque);
        deque.poll();
      }
    }
  }

  public int getDequeueSize() {
    return deque.size();
  }

  @Override // from MonteCarloTrial
  public QsaInterface qsaInterface() {
    return qsa();
  }

  public Policy getPolicy() {
    return policy;
  }
}
