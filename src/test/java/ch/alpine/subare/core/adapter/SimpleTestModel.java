// code by jph
package ch.alpine.subare.core.adapter;

import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.core.StandardModel;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.num.Boole;
import ch.alpine.tensor.red.KroneckerDelta;

/** states = 0 1 2 with start = 0 and terminal = 2
 * 
 * only single action is possible */
// DO NOT MODIFY THIS IMPLEMENTATION
// MODIFY AND ADAPT A COPY IF NEEDED
public enum SimpleTestModel implements MonteCarloInterface, StandardModel {
  INSTANCE;

  private static final Scalar TWO = RealScalar.of(2);

  @Override // from DiscreteModel
  public Tensor states() {
    return Range.of(0, 3);
  }

  @Override // from DiscreteModel
  public Tensor actions(Tensor state) {
    return Tensors.of(Boole.of(!isTerminal(state)));
  }

  @Override // from DiscreteModel
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  // ---
  @Override
  public Tensor move(Tensor state, Tensor action) {
    return state.add(action);
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    return expectedReward(state, action);
  }

  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return state.equals(TWO);
  }

  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return Tensors.of(RealScalar.ZERO);
  }

  @Override // from ActionValueInterface
  public Scalar expectedReward(Tensor state, Tensor action) {
    return TWO.subtract(state);
  }

  @Override // from TransitionInterface
  public Tensor transitions(Tensor state, Tensor action) {
    return Tensors.of(move(state, action));
  }

  @Override // from TransitionInterface
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    return KroneckerDelta.of(move(state, action), next);
  }
}
