// code by jph
package ch.alpine.subare.ch06.windy;

import ch.alpine.subare.core.alg.ActionValueIterations;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.tensor.RealScalar;

enum WindygridHelper {
  ;
  static DiscreteQsa getOptimalQsa(Windygrid windygrid) {
    return ActionValueIterations.solve(windygrid, RealScalar.of(.0001));
  }
}
