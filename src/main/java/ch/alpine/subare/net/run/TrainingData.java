// code by jph
package ch.alpine.subare.net.run;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.Integers;

public record TrainingData(Tensor xdata, Tensor ydata) {
  public TrainingData {
    Integers.requireEquals(xdata.length(), ydata.length());
  }
}
