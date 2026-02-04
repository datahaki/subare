// code by jph
package ch.alpine.subare.book.ch06.walk;

import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.EGreedyPolicy;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.Policies;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.sca.Round;

/** <pre>
 * {0, 0} 0
 * {1, 0} 0.24
 * {2, 0} 0.41
 * {3, 0} 0.54
 * {4, 0} 0.74
 * {5, 0} 0.87
 * {6, 0} 0
 * </pre> */
enum MCES_Randomwalk {
  ;
  static void main() {
    Randomwalk randomwalk = new Randomwalk(5);
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(randomwalk);
    StateActionCounter sac = new DiscreteStateActionCounter();
    EGreedyPolicy policy = (EGreedyPolicy) PolicyType.EGREEDY.bestEquiprobable(randomwalk, mces.qsa(), sac);
    int batches = 1000;
    for (int count = 0; count < batches; ++count) {
      if (count == 0) {
        boolean equals = Policies.equals(randomwalk, policy, EquiprobablePolicy.create(randomwalk));
        if (!equals)
          throw new RuntimeException();
      }
      ExploringStarts.batch(randomwalk, policy, mces);
    }
    DiscreteQsa qsa = mces.qsa();
    DiscreteUtils.print(qsa, Round._2);
  }
}
