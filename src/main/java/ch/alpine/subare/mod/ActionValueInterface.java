// code by jph
package ch.alpine.subare.mod;

import ch.alpine.subare.val.VsInterface;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public interface ActionValueInterface {
  /** @param state
   * @param action
   * @return all states that are a possible result of taking action in given state */
  Tensor transitions(Tensor state, Tensor action);

  /** @param state
   * @param action
   * @param next
   * @return probability to reach next as a result of taking action in given state */
  Scalar transitionProbability(Tensor state, Tensor action, Tensor next);

  /** @param state
   * @param action
   * @return expected reward when action is taken in state */
  Scalar expectedReward(Tensor state, Tensor action);

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
  default Scalar qsa(Tensor state, Tensor action, VsInterface gvalues) {
    Scalar sum = transitions(state, action).stream() //
        .map(next -> transitionProbability(state, action, next).multiply(gvalues.value(next))) //
        .reduce(Scalar::add) //
        .orElseThrow();
    return expectedReward(state, action).add(sum);
  }
}
