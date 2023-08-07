// code by jph
package ch.alpine.subare.ch06.maxbias;

import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.core.StandardModel;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.Throw;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.red.KroneckerDelta;

/** Example 6.7 p.134: Maximization bias
 * 
 * Credit: Hado van Hasselt (2010, 2011) */
public class Maxbias implements StandardModel, MonteCarloInterface {
  static final Scalar MEAN = RealScalar.of(-0.1);
  static final Scalar STATE_A = RealScalar.of(2);
  static final Scalar STATE_B = RealScalar.of(1);
  static final Scalar STATE_L = RealScalar.of(0);
  final Tensor states = Tensors.vector(0, 1, 2, 3).unmodifiable();
  final Tensor actionsA = Tensors.vector(-1, 1); // left, or right
  final Tensor actionsB;
  final Distribution distribution = NormalDistribution.of(MEAN, RealScalar.ONE);

  public Maxbias(int choices) {
    actionsB = Range.of(0, choices).unmodifiable();
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    if (state.equals(STATE_A))
      return actionsA;
    if (state.equals(STATE_B))
      return actionsB;
    return Tensors.vector(0);
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  // ---
  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (state.equals(STATE_A))
      return state.add(action);
    if (state.equals(STATE_B))
      return STATE_L;
    return state;
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    return state.equals(STATE_B) //
        ? RandomVariate.of(distribution)
        : RealScalar.ZERO;
  }

  // ---
  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return !state.equals(STATE_A) //
        && !state.equals(STATE_B);
  }

  // ---
  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return Tensors.of(STATE_A);
  }

  // ---
  @Override
  public Scalar expectedReward(Tensor state, Tensor action) {
    return state.equals(STATE_B) //
        ? MEAN
        : RealScalar.ZERO;
  }

  @Override
  public Tensor transitions(Tensor state, Tensor action) {
    return Tensors.of(move(state, action));
  }

  @Override
  public Scalar transitionProbability(Tensor state, Tensor action, Tensor next) {
    // TODO SUBARE this implementation does not make sense
    if (move(state, action).equals(next))
      return KroneckerDelta.of(move(state, action), next);
    throw new Throw(state, action, next);
  }
}
