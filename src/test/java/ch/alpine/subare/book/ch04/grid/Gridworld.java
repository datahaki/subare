// code by jph
package ch.alpine.subare.book.ch04.grid;

import java.util.function.Predicate;

import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.util.DeterministicStandardModel;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

/** Example 4.1, p.76 */
public class Gridworld extends DeterministicStandardModel implements MonteCarloInterface {
  private static final Scalar NEGATIVE_ONE = RealScalar.ONE.negate();
  private static final Tensor TERMINATE1 = Tensors.vector(0, 0); // A
  private static final Tensor TERMINATE2 = Tensors.vector(3, 3); // A'
  private static final Clip CLIP = Clips.positive(3);
  private static final Tensor ACTIONS = Tensors.matrix(new Number[][] { //
      { -1, 0 }, //
      { +1, 0 }, //
      { 0, -1 }, //
      { 0, +1 } //
  }).unmodifiable();
  static final int NX = 4;
  static final int NY = 4;
  // ---
  private static final Tensor STATES = Flatten.of(Array.of(Tensors::vector, NX, NY), 1).unmodifiable();

  @Override
  public Tensor states() {
    return STATES;
  }

  @Override
  public Tensor actions(Tensor state) {
    return ACTIONS;
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  // ---
  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor stateS) {
    return isTerminal(state) //
        ? RealScalar.ZERO
        : NEGATIVE_ONE;
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    return isTerminal(state) //
        ? state
        : state.add(action).maps(CLIP);
  }

  // ---
  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return Tensor.of(STATES.stream().filter(Predicate.not(this::isTerminal)));
  }

  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return state.equals(TERMINATE1) //
        || state.equals(TERMINATE2);
  }
}
