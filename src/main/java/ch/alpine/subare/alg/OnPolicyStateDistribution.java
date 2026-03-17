// code by jph
package ch.alpine.subare.alg;

import ch.alpine.subare.api.mod.StandardModel;
import ch.alpine.subare.api.pol.Policy;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** box in 9.2, p.161
 * 
 * output is denoted with eta: S -> R */
public record OnPolicyStateDistribution(StandardModel standardModel, Policy policy) {
  public DiscreteVs iterate(DiscreteVs vs_old) {
    DiscreteVs vs_new = vs_old.discounted(RealScalar.ZERO);
    for (Tensor state : standardModel.states()) {
      for (Tensor action : standardModel.actions(state)) {
        Scalar pi = policy.probability(state, action);
        Scalar value = vs_old.value(state);
        for (Tensor next : standardModel.transitions(state, action)) {
          Scalar prob = standardModel.transitionProbability(state, action, next);
          vs_new.increment(next, pi.multiply(prob).multiply(value));
        }
      }
    }
    return vs_new;
  }
}
