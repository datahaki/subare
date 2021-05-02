// code by jph
package ch.ethz.idsc.subare.ch02.bandits2;

import ch.alpine.tensor.RealScalar;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;

/* package */ enum BanditsHelper {
  ;
  static DiscreteQsa getOptimalQsa(StandardModel standardModel) {
    return ActionValueIterations.solve(standardModel, RealScalar.of(.0001));
  }
}
