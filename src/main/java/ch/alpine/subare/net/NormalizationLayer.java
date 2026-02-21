// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public record NormalizationLayer(Tensor mean, Scalar std) implements Layer {
  @Override
  public Tensor forward(Tensor x) {
    return x.subtract(mean).divide(std);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return gradOutput.divide(std);
  }

  @Override
  public void update() {
  }

  @Override
  public Tensor error(Tensor y) {
    throw new IllegalStateException();
  }
}
