// code by jph
package ch.alpine.subare.ch02.bandits2;

import ch.alpine.subare.core.alg.ActionValueIterations;
import ch.alpine.subare.core.api.StandardModel;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.tensor.RealScalar;

/* package */ enum BanditsHelper {
  ;
  static DiscreteQsa getOptimalQsa(StandardModel standardModel) {
    return ActionValueIterations.solve(standardModel, RealScalar.of(.0001));
  }
}
