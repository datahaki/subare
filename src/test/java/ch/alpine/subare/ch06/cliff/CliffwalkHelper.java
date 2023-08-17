// code by jph
package ch.alpine.subare.ch06.cliff;

import ch.alpine.subare.core.alg.ActionValueIterations;
import ch.alpine.subare.core.alg.ValueIteration;
import ch.alpine.subare.core.api.Policy;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.PolicyType;
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
