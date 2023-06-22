// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.ch04.gambler;

import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.alg.ValueIteration;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteUtils;
import ch.alpine.subare.core.util.DiscreteValueFunctions;
import ch.alpine.subare.core.util.DiscreteVs;
import ch.alpine.subare.core.util.Policies;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.sca.N;

/** Shangtong Zhang states that using double precision in python
 * "due to tie and precision, can't reproduce the optimal policy in book"
 * 
 * Unlike stated in the book, there is not a unique optimal policy but many
 * using symbolic expressions we can reproduce the policy in book and
 * all other optimal actions
 * 
 * chapter 4, example 3 */
/* package */ enum VI_Gambler {
  ;
  public static void main(String[] args) {
    GamblerModel gamblerModel = GamblerModel.createDefault();
    DiscreteQsa ref = GamblerHelper.getOptimalQsa(gamblerModel);
    ValueIteration vi = new ValueIteration(gamblerModel, gamblerModel);
    vi.untilBelow(RealScalar.of(1e-20));
    final DiscreteVs vs = vi.vs();
    final DiscreteVs vr = DiscreteUtils.createVs(gamblerModel, ref);
    Scalar diff = DiscreteValueFunctions.distance(vs, vr);
    System.out.println("error=" + N.DOUBLE.apply(diff));
    Policy policy = PolicyType.GREEDY.bestEquiprobable(gamblerModel, ref, null);
    Policies.print(policy, gamblerModel.states());
  }
}
