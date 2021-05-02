// code by jph
package ch.ethz.idsc.subare.ch06.walk;

import ch.alpine.tensor.sca.Round;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteStateActionCounter;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.EquiprobablePolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.Policies;
import ch.ethz.idsc.subare.core.util.PolicyType;

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
  public static void main(String[] args) throws Exception {
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
