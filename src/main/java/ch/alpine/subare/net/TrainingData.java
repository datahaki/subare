// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.Integers;

public record TrainingData(Tensor xdata, Tensor ydata) {
  public TrainingData(Tensor xdata, Tensor ydata) {
    Integers.requireEquals(xdata.length(), ydata.length());
    this.xdata = xdata;
    this.ydata = ydata;
  }
}
