// code by jph
package ch.alpine.subare.ch04.grid;

import ch.alpine.subare.core.alg.ActionValueIterations;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.tensor.RealScalar;

enum GridworldHelper {
  ;
  static DiscreteQsa getOptimalQsa(Gridworld gridworld) {
    return ActionValueIterations.solve(gridworld, RealScalar.of(.0001));
  }
}
