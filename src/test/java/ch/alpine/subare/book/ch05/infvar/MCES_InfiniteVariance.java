// code by jph
package ch.alpine.subare.book.ch05.infvar;

import ch.alpine.subare.api.Policy;
import ch.alpine.subare.mc.MonteCarloExploringStarts;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.EquiprobablePolicy;
import ch.alpine.subare.util.ExploringStarts;

enum MCES_InfiniteVariance {
  ;
  static void main() {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(infiniteVariance);
    Policy policy = EquiprobablePolicy.create(infiniteVariance);
    for (int c = 0; c < 100; ++c)
      ExploringStarts.batch(infiniteVariance, policy, mces);
    DiscreteQsa qsa = mces.qsa();
    DiscreteUtils.print(qsa);
  }
}
