// code by jph
package ch.alpine.subare.math;

import java.util.Objects;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.itp.BinaryAverage;
import ch.alpine.tensor.itp.LinearBinaryAverage;

/** consumes Tensor/Scalar and tracks average
 * without the overhead of storing all input. */
public class AverageTracker {
  private final BinaryAverage binaryAverage;
  private Tensor average = null;
  private Scalar count = RealScalar.ZERO;

  public AverageTracker(BinaryAverage binaryAverage) {
    this.binaryAverage = Objects.requireNonNull(binaryAverage);
  }

  public AverageTracker() {
    this(LinearBinaryAverage.INSTANCE);
  }

  /** @param tensor that contributes to the average of all tracked {@link Tensor}s */
  public void track(Tensor tensor) {
    if (Scalars.isZero(count))
      average = tensor.copy();
    count = count.add(RealScalar.ONE);
    average = binaryAverage.split(average, tensor, count.reciprocal());
  }

  /** @return average of {@link Tensor}s tracked by {@link #track(Tensor)},
   * or null if function {@link #track(Tensor)} has not been called. */
  public Tensor get() {
    return average;
  }

  /** @return {@link #get()} cast to {@link Scalar} */
  public Scalar Get() {
    return (Scalar) average;
  }

  @Override // from Object
  public String toString() {
    return MathematicaFormat.concise("AverageTracker", average, count);
  }
}
