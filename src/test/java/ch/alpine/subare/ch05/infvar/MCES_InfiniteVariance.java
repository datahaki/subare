// code by jph
package ch.alpine.subare.ch05.infvar;

import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.mc.MonteCarloExploringStarts;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteUtils;
import ch.alpine.subare.core.util.EquiprobablePolicy;
import ch.alpine.subare.core.util.ExploringStarts;

enum MCES_InfiniteVariance {
  ;
  public static void main(String[] args) {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(infiniteVariance);
    Policy policy = EquiprobablePolicy.create(infiniteVariance);
    for (int c = 0; c < 100; ++c)
      ExploringStarts.batch(infiniteVariance, policy, mces);
    DiscreteQsa qsa = mces.qsa();
    DiscreteUtils.print(qsa);
  }
}
