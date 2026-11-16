// code by jph
package ch.alpine.subare.net.run;

import java.util.Objects;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.qty.Timing;

public class TimeoutListener implements NetTrainListener {
  private final Scalar max_duration;
  private Timing timing;

  public TimeoutListener(Scalar max_duration) {
    this.max_duration = max_duration;
  }

  @Override
  public boolean epoch(NetTrain netTrain) {
    if (Objects.isNull(timing))
      timing = Timing.started();
    return Scalars.lessThan(max_duration, timing.seconds());
  }
}
