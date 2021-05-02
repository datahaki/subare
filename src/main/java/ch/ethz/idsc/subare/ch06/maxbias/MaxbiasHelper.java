// code by jph
package ch.ethz.idsc.subare.ch06.maxbias;

import ch.alpine.tensor.RealScalar;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;

enum MaxbiasHelper {
  ;
  static DiscreteQsa getOptimalQsa(Maxbias maxbias) {
    return ActionValueIterations.solve(maxbias, RealScalar.of(.0001));
  }
}
