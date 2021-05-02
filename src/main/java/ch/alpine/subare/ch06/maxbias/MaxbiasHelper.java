// code by jph
package ch.alpine.subare.ch06.maxbias;

import ch.alpine.subare.core.alg.ActionValueIterations;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.tensor.RealScalar;

enum MaxbiasHelper {
  ;
  static DiscreteQsa getOptimalQsa(Maxbias maxbias) {
    return ActionValueIterations.solve(maxbias, RealScalar.of(.0001));
  }
}
