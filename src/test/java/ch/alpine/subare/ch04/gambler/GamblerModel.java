// code by jph
// inspired by Shangtong Zhang
package ch.alpine.subare.ch04.gambler;

import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.core.StandardModel;
import ch.alpine.subare.util.Coinflip;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Throw;
import ch.alpine.tensor.alg.Last;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.red.KroneckerDelta;
import ch.alpine.tensor.red.Min;

/** Example 4.3 p.84: Gambler's problem
 * an action defines the amount of coins to bet
 * the action has to be non-zero unless the capital == 0
 * or the terminal cash has been reached
 * 
 * [no further references are provided in the book] */
public class GamblerModel implements StandardModel, MonteCarloInterface {
  private final Tensor states;
  private final Scalar last;
  private final Scalar P_win;
  private final Coinflip coinflip;

  public static GamblerModel createDefault() {
    return new GamblerModel(100, RationalScalar.of(4, 10));
  }

  /** @param max stake
   * @param P_win probabilty of winning a coin toss */
  public GamblerModel(int max, Scalar P_win) {
    states = Range.of(0, max + 1).unmodifiable();
    last = Last.of(states);
    this.P_win = P_win;
    coinflip = Coinflip.of(P_win);
  }

  @Override
  public Tensor states() {
    return states;
  }

  /** @return possible stakes */
  @Override
  public Tensor actions(Tensor state) {
    if (isTerminal(state))
      return Tensors.of(RealScalar.ZERO);
    // here we deviate from the book and the code by STZ:
    // we require that the bet=action is non-zero,
    // if the state is non-terminal, 0 < cash < 100.
    // otherwise the player can stall (the iteration) forever.
    Scalar stateS = (Scalar) state;
    return Range.of(1, Scalars.intValueExact(Min.of(stateS, last.subtract(stateS))) + 1);
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  // ---
  @Override
  public Tensor move(Tensor state, Tensor action) { // non-deterministic
    if (coinflip.tossHead()) // win
      return state.add(action);
    return state.subtract(action);
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) { // deterministic
    return isTerminal(state) //
        ? RealScalar.ZERO
        : KroneckerDelta.of(next, last);
  }

  // ---
  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return states.extract(1, states.length() - 1);
  }

  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return state.equals(RealScalar.ZERO) //
        || state.equals(last);
  }

  // ---
  @Override
  public Scalar expectedReward(Tensor state, Tensor action) {
    return isTerminal(state) //
        ? RealScalar.ZERO
        : KroneckerDelta.of(state.add(action), last).multiply(P_win); // P_win * 1, or 0
  }

  @Override
  public Tensor transitions(Tensor state, Tensor action) {
    return isTerminal(state) //
        ? Tensors.of(state)
        : Tensors.of( //
            state.add(action), // with probability P_win
            state.subtract(action)); // with probability 1 - P_win
  }

  @Override
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    if (isTerminal(state))
      return RealScalar.ONE;
    if (state.add(action).equals(next))
      return P_win;
    if (state.subtract(action).equals(next))
      return RealScalar.ONE.subtract(P_win);
    throw new Throw(state, action, next);
  }
}
