// code by jph
package ch.alpine.subare.book.ch06.cliff;

import ch.alpine.subare.alg.ActionValueIterations;
import ch.alpine.subare.alg.ValueIteration;
import ch.alpine.subare.api.Policy;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.tensor.RealScalar;

enum CliffwalkHelper {
  ;
  static DiscreteQsa getOptimalQsa(Cliffwalk cliffwalk) {
    return ActionValueIterations.solve(cliffwalk, RealScalar.of(.0001));
  }

  static Policy getOptimalPolicy(Cliffwalk cliffwalk) {
    ValueIteration vi = new ValueIteration(cliffwalk, cliffwalk);
    vi.untilBelow(RealScalar.of(1e-10));
    return PolicyType.GREEDY.bestEquiprobable(cliffwalk, vi.vs(), null);
  }
}
