// inspired by discussion with fabio
// code by jph
package ch.alpine.subare.demo.fish;

import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.core.adapter.DeterministicStandardModel;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;
import ch.alpine.tensor.sca.Round;

class Fishfarm extends DeterministicStandardModel implements MonteCarloInterface {
  public final Scalar TERMINAL_LIMIT;
  public final int max_fish;
  public final Clip clip;
  public final int period;
  private final Tensor states;

  public Fishfarm(int period, int max_fish) {
    this.period = period;
    this.max_fish = max_fish;
    TERMINAL_LIMIT = RealScalar.of(max_fish * 0.9);
    clip = Clips.interval(1, max_fish);
    states = Flatten.of(Array.of(Tensors::vector, period + 1, max_fish + 1), 1).unmodifiable();
    // states.append(Tensors.vector(0));
    // states.append(Tensors.vector(period, 1));
  }

  @Override
  public Tensor states() {
    return states.unmodifiable();
  }

  @Override
  public Tensor actions(Tensor state) {
    Scalar fish = state.Get(1);
    return Range.of(0, Scalars.intValueExact(fish) + 1);
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  // ---
  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    // TODO SUBARE can make probabilistic: binomial
    Scalar time = state.Get(0).add(RealScalar.ONE);
    Scalar fish = state.Get(1).subtract(action); // after fishing
    fish = fish.add(growth(fish));
    fish = clip.apply(fish); // TODO SUBARE this should not be necessary
    return Tensors.of(time, fish);
  }

  public Scalar growth(Scalar population) {
    Scalar max = RealScalar.of(max_fish);
    Scalar max_half = max.multiply(RealScalar.of(0.5)); // n/2
    Scalar factor = RealScalar.of(8).divide(max);
    Scalar x = population.subtract(max_half).multiply(factor); // x-n/2
    Scalar res = DLogisticSigmoid.FUNCTION.apply(x);
    return Round.FUNCTION.apply(res.multiply(max));
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    if (isTerminal(state))
      return RealScalar.ZERO;
    Scalar fish = (Scalar) action;
    if (!isTerminal(next))
      return fish;
    Scalar remain = next.Get(1);
    Scalar penalty = Scalars.lessEquals(TERMINAL_LIMIT, remain) //
        ? RealScalar.ZERO
        : RealScalar.of(-1000);
    return fish.add(penalty);
  }

  // ---
  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return state.Get(0).equals(RealScalar.of(period));
  }

  @Override // from MonteCarloInterface
  public Tensor startStates() {
    return Tensor.of(states().stream() //
        .filter(state -> Scalars.isZero(state.Get(0))));
  }
}
