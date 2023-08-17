// code by jph
package ch.alpine.subare.book.ch06.windy;

import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.util.DeterministicStandardModel;
import ch.alpine.subare.util.StateActionMap;
import ch.alpine.subare.util.StateActionMaps;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

/** Example 6.5 p.130: Windy Gridworld
 * 
 * Credit: Tom Kalt */
public class Windygrid extends DeterministicStandardModel implements MonteCarloInterface {
  static final Scalar NEGATIVE_ONE = RealScalar.ONE.negate();
  static final int NX = 10;
  static final int NY = 7;
  static final Tensor START = Tensors.vector(0, 3);
  static final Tensor GOAL = Tensors.vector(7, 3).unmodifiable();
  private static final Tensor WIND = Tensors.vector(0, 0, 0, 1, 1, 1, 2, 2, 1, 0).negate();
  private static final Clip CLIP_X = Clips.positive(9);
  private static final Clip CLIP_Y = Clips.positive(6);
  // ---
  private final Tensor states = Flatten.of(Array.of(Tensors::vector, NX, NY), 1).unmodifiable();
  final Tensor actions;
  private final StateActionMap stateActionMap;

  public static Windygrid createFour() {
    Tensor actions = Tensors.matrix(new Number[][] { //
        { 0, -1 }, //
        { 0, +1 }, //
        { -1, 0 }, //
        { +1, 0 } //
    }).unmodifiable();
    return new Windygrid(actions);
  }

  public static Windygrid createKing() {
    Tensor actions = Tensors.matrix(new Number[][] { //
        { 0, -1 }, //
        { 0, +1 }, //
        { -1, 0 }, //
        { +1, 0 }, //
        // ---
        { +1, -1 }, //
        { +1, +1 }, //
        { -1, -1 }, //
        { -1, +1 } //
    }).unmodifiable();
    return new Windygrid(actions);
  }

  private Windygrid(Tensor actions) {
    this.actions = actions.unmodifiable();
    stateActionMap = StateActionMaps.build(states, actions, this);
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return stateActionMap.actions(state);
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE; // undiscounted task
  }

  // ---
  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    return isTerminal(next) //
        ? RealScalar.ZERO
        : NEGATIVE_ONE; // -1 until goal is reached
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    // wind is added first
    Tensor next = state.copy();
    int x = Scalars.intValueExact(next.Get(0)); // wind depends on x coordinate
    next.set(scalar -> scalar.add(WIND.Get(x)), 1); // wind shift in y coordinate
    next.set(CLIP_Y, 1);
    next = next.add(action);
    next.set(CLIP_X, 0);
    next.set(CLIP_Y, 1);
    return next;
  }

  // ---
  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return Tensors.of(START);
  }

  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return state.equals(GOAL);
  }
}
