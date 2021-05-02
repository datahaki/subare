// code by jph
package ch.alpine.subare.demo.bus;

import java.awt.Dimension;

import ch.alpine.subare.core.TerminalInterface;
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

/* package */ class Charger extends DeterministicStandardModel implements TerminalInterface {
  private final TripProfile tripProfile;
  private final Clip clipCapacity;
  private final Tensor states;
  private final Tensor actions = Range.of(0, 5).unmodifiable();
  public final Dimension dimension;

  public Charger(TripProfile tripProfile, int capacity) {
    this.tripProfile = tripProfile;
    states = Flatten.of(Array.of(Tensors::vector, tripProfile.length(), capacity), 1).unmodifiable();
    clipCapacity = Clips.positive(capacity - 1);
    dimension = new Dimension(tripProfile.length(), capacity);
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    return actions;
  }

  @Override
  public Scalar gamma() {
    return RealScalar.ONE;
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    final int time = Scalars.intValueExact(state.Get(0));
    Tensor next = state.copy();
    next.set(RealScalar.ONE::add, 0);
    Scalar drawn = tripProfile.unitsDrawn(time);
    next.set(capacity -> capacity.add(action).subtract(drawn), 1);
    next.set(clipCapacity, 1);
    return next;
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    final int capacity = Scalars.intValueExact(state.Get(1));
    if (isTerminal(next)) {
      if (isTerminal(state))
        return RealScalar.ZERO;
      return 0 == capacity //
          ? RealScalar.of(-10)
          : RealScalar.ZERO;
    }
    final int time = Scalars.intValueExact(state.Get(0));
    Scalar total = tripProfile.costPerUnit(time).multiply((Scalar) action).negate();
    if (capacity == 0)
      total = total.add(RealScalar.of(-20)); // TODO possibly make terminal
    return total;
  }

  @Override // from TerminalInterface
  public boolean isTerminal(Tensor state) {
    return Scalars.intValueExact(state.Get(0)) == tripProfile.length() - 1;
  }
}
