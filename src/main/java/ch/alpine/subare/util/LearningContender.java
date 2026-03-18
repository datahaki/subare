// code by jph
package ch.alpine.subare.util;

import ch.alpine.subare.mod.MonteCarloInterface;
import ch.alpine.subare.pol.EGreedyPolicy;
import ch.alpine.subare.pol.PolicyType;
import ch.alpine.subare.rate.ExplorationRate;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.val.DiscreteQsa;

/**  */
public record LearningContender(MonteCarloInterface monteCarloInterface, Sarsa sarsa, DiscreteQsa qsa) {
  /** @param monteCarloInterface
   * @param sarsa
   * @return */
  public static LearningContender sarsa(MonteCarloInterface monteCarloInterface, Sarsa sarsa) {
    return new LearningContender(monteCarloInterface, sarsa, sarsa.qsa());
  }

  // ---
  public void stepAndCompare(ExplorationRate explorationRate, int nstep, DiscreteQsa ref) {
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, ref, sarsa.sac());
    policy.setExplorationRate(explorationRate);
    ExploringStarts.batch(monteCarloInterface, policy, nstep, sarsa);
  }

  public Infoline infoline(DiscreteQsa ref) {
    return Infoline.of(monteCarloInterface, ref, qsa);
  }
}
