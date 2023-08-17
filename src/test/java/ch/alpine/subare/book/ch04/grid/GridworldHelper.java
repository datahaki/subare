// code by jph
package ch.alpine.subare.book.ch04.grid;

import ch.alpine.subare.alg.ActionValueIterations;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.tensor.RealScalar;

enum GridworldHelper {
  ;
  static DiscreteQsa getOptimalQsa(Gridworld gridworld) {
    return ActionValueIterations.solve(gridworld, RealScalar.of(.0001));
  }
}
