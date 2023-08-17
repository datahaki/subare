// code by jph
package ch.alpine.subare.book.ch02.bandits2;

import ch.alpine.subare.alg.ActionValueIterations;
import ch.alpine.subare.api.StandardModel;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.tensor.RealScalar;

/* package */ enum BanditsHelper {
  ;
  static DiscreteQsa getOptimalQsa(StandardModel standardModel) {
    return ActionValueIterations.solve(standardModel, RealScalar.of(.0001));
  }
}
