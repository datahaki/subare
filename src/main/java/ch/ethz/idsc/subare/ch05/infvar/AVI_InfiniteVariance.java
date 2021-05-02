// code by jph
package ch.ethz.idsc.subare.ch05.infvar;

import ch.alpine.tensor.RealScalar;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;

enum AVI_InfiniteVariance {
  ;
  public static void main(String[] args) {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    DiscreteQsa qsa = ActionValueIterations.solve(infiniteVariance, RealScalar.of(.00001));
    DiscreteUtils.print(qsa);
  }
}
