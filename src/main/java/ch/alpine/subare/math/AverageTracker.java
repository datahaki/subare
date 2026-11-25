// code by jph
package ch.alpine.subare.math;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.red.MeanSummaryStatistics;

/** consumes Tensor/Scalar and tracks average
 * without the overhead of storing all input. */
public class AverageTracker {
  private final MeanSummaryStatistics mss = new MeanSummaryStatistics();

  /** @param tensor that contributes to the average of all tracked {@link Tensor}s */
  public void digest(Tensor tensor) {
    mss.accept(tensor);
  }

  /** @return average of {@link Tensor}s tracked by {@link #digest(Tensor)},
   * or null if function {@link #digest(Tensor)} has not been called. */
  public Tensor get() {
    return mss.getMean().orElse(null);
  }

  /** @return {@link #get()} cast to {@link Scalar} */
  public Scalar Get() {
    return (Scalar) mss.getMean().orElse(null);
  }

  @Override // from Object
  public String toString() {
    return mss.toString();
  }
}
