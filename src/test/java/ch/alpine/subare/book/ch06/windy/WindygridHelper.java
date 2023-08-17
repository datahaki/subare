// code by jph
package ch.alpine.subare.book.ch06.windy;

import ch.alpine.subare.alg.ActionValueIterations;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.tensor.RealScalar;

enum WindygridHelper {
  ;
  static DiscreteQsa getOptimalQsa(Windygrid windygrid) {
    return ActionValueIterations.solve(windygrid, RealScalar.of(.0001));
  }
}
