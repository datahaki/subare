// code by jph
package ch.alpine.subare.demo.bus;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Range;

/* package */ class ConstantDrawTrip implements TripProfile {
  private final int length;
  private final Tensor costPerUnit;
  private final Scalar draw;

  public ConstantDrawTrip(int length, int amount) {
    this.length = length;
    costPerUnit = Range.of(0, length).map(new Sawtooth(3)); // .map(Increment.ONE);
    draw = RealScalar.of(amount);
  }

  @Override
  public int length() {
    return length;
  }

  @Override
  public Scalar costPerUnit(int index) {
    return costPerUnit.Get(index);
  }

  @Override
  public Scalar unitsDrawn(int time) {
    return draw;
  }
}
