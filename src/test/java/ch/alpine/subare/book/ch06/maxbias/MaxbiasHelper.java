// code by jph
package ch.alpine.subare.book.ch06.maxbias;

import ch.alpine.subare.alg.ActionValueIterations;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.tensor.RealScalar;

enum MaxbiasHelper {
  ;
  static DiscreteQsa getOptimalQsa(Maxbias maxbias) {
    return ActionValueIterations.solve(maxbias, RealScalar.of(.0001));
  }
}
