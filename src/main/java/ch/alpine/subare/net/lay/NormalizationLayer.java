// code by jph
package ch.alpine.subare.net.lay;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public record NormalizationLayer(Tensor mean, Scalar std) implements NetLayer {
  @Override
  public Tensor forward(Tensor x) {
    return x.subtract(mean).divide(std);
  }

  @Override
  public Tensor forwardBatch(Tensor x) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return gradOutput.divide(std);
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    throw new UnsupportedOperationException();
  }
}
