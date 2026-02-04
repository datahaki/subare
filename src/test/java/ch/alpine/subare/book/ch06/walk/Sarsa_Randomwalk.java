// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.book.ch06.walk;

import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.PolicyBase;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.sca.Round;

/** determines state action value function q(s, a).
 * initial policy is irrelevant because each state allows only one action.
 * 
 * <pre>
 * {0, 0} 0
 * {1, 0} 0.16
 * {2, 0} 0.35
 * {3, 0} 0.47
 * {4, 0} 0.59
 * {5, 0} 0.79
 * {6, 0} 0
 * </pre> */
enum Sarsa_Randomwalk {
  ;
  static void handle(SarsaType sarsaType, int nstep) {
    System.out.println(sarsaType);
    Randomwalk randomwalk = new Randomwalk(5);
    DiscreteQsa qsa = DiscreteQsa.build(randomwalk);
    LearningRate learningRate = DefaultLearningRate.of(2, 0.6);
    StateActionCounter sac = new DiscreteStateActionCounter();
    PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(randomwalk, qsa, sac);
    Sarsa sarsa = sarsaType.sarsa(randomwalk, learningRate, qsa, sac, policy);
    Policy policyEqui = EquiprobablePolicy.create(randomwalk);
    for (int count = 0; count < 1000; ++count)
      ExploringStarts.batch(randomwalk, policyEqui, nstep, sarsa);
    DiscreteUtils.print(qsa, Round._2);
  }

  static void main() {
    for (SarsaType type : SarsaType.values())
      handle(type, 1);
  }
}
