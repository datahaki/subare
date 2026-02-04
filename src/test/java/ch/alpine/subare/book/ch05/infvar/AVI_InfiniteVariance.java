// code by jph
package ch.alpine.subare.book.ch05.infvar;

import ch.alpine.subare.alg.ActionValueIterations;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.tensor.RealScalar;

enum AVI_InfiniteVariance {
  ;
  static void main() {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    DiscreteQsa qsa = ActionValueIterations.solve(infiniteVariance, RealScalar.of(.00001));
    DiscreteUtils.print(qsa);
  }
}
