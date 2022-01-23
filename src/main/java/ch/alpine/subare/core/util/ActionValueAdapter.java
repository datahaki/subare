// code by jph
package ch.alpine.subare.core.util;

import ch.alpine.subare.core.ActionValueInterface;
import ch.alpine.subare.core.VsInterface;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public record ActionValueAdapter(ActionValueInterface actionValueInterface) {
  /** function implements the formula
   * Sum_{s', r} p(s', r | s, a) * [r + gamma * v_*(s')]
   * 
   * general term in bellman equation:
   * Sum_{s', r} p(s', r | s, a) * (r + gamma * v_pi(s'))
   * 
   * where
   * v_*(s) == max_a q_*(s, a)
   * 
   * general term in bellman equation:
   * Sum_{s', r} p(s', r | s, a) * (r + gamma * v_pi(s'))
   * for deterministic move and reward the formula simplifies to
   * 1 * (r + gamma * v_pi(s'))
   * 
   * @param state
   * @param action
   * @param gvalues value function already discounted by gamma
   * @return expected return for the best action for that state */
  public Scalar qsa(Tensor state, Tensor action, VsInterface gvalues) {
    Scalar sum = actionValueInterface.transitions(state, action).stream() //
        .map(next -> actionValueInterface.transitionProbability(state, action, next).multiply(gvalues.value(next))) //
        .reduce(Scalar::add) //
        .orElseThrow();
    return actionValueInterface.expectedReward(state, action).add(sum);
  }
}
