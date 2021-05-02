// code by jph
package ch.ethz.idsc.subare.ch04.grid;

import ch.alpine.tensor.RealScalar;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;

enum GridworldHelper {
  ;
  static DiscreteQsa getOptimalQsa(Gridworld gridworld) {
    return ActionValueIterations.solve(gridworld, RealScalar.of(.0001));
  }
}
