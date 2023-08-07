// code by jph
package ch.alpine.subare.ch05.infvar;

import ch.alpine.subare.core.alg.ActionValueIterations;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteUtils;
import ch.alpine.tensor.RealScalar;

enum AVI_InfiniteVariance {
  ;
  public static void main(String[] args) {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    DiscreteQsa qsa = ActionValueIterations.solve(infiniteVariance, RealScalar.of(.00001));
    DiscreteUtils.print(qsa);
  }
}
