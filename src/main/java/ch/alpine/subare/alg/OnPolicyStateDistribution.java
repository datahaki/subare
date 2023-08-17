// code by jph
package ch.alpine.subare.alg;

import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.StateActionModel;
import ch.alpine.subare.api.TransitionInterface;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** box in 9.2, p.161
 * 
 * output is denoted with eta: S -> R */
public record OnPolicyStateDistribution( //
    StateActionModel stateActionModel, //
    TransitionInterface transitionInterface, //
    Policy policy) {
  public DiscreteVs iterate(DiscreteVs vs_old) {
    DiscreteVs vs_new = vs_old.discounted(RealScalar.ZERO);
    for (Tensor state : stateActionModel.states()) {
      for (Tensor action : stateActionModel.actions(state)) {
        Scalar pi = policy.probability(state, action);
        Scalar value = vs_old.value(state);
        for (Tensor next : transitionInterface.transitions(state, action)) {
          Scalar prob = transitionInterface.transitionProbability(state, action, next);
          vs_new.increment(next, pi.multiply(prob).multiply(value));
        }
      }
    }
    return vs_new;
  }
}
