// code by jph
package ch.ethz.idsc.subare.demo.fish;

import ch.alpine.tensor.RealScalar;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;

enum FishfarmHelper {
  ;
  static DiscreteQsa getOptimalQsa(Fishfarm cliffwalk) {
    return ActionValueIterations.solve(cliffwalk, RealScalar.of(.0001));
  }
  // static Policy getOptimalPolicy(Fishfarm cliffwalk) {
  // ValueIteration vi = new ValueIteration(cliffwalk, cliffwalk);
  // vi.untilBelow(RealScalar.of(1e-10));
  // return GreedyPolicy.bestEquiprobable(cliffwalk, vi.vs());
  // }
}
