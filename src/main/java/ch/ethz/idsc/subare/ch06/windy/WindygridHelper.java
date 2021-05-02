// code by jph
package ch.ethz.idsc.subare.ch06.windy;

import ch.alpine.tensor.RealScalar;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;

enum WindygridHelper {
  ;
  static DiscreteQsa getOptimalQsa(Windygrid windygrid) {
    return ActionValueIterations.solve(windygrid, RealScalar.of(.0001));
  }
}
