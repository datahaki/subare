// code by jph
package ch.alpine.subare.net.lay;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.io.MathematicaFormat;

public class ScalingLayer extends SimpleElementwiseLayer {
  private final Scalar scalar;

  public ScalingLayer(Scalar scalar) {
    this.scalar = scalar;
  }

  @Override
  public Scalar func(Scalar x) {
    return x.multiply(scalar);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return gradOutput.multiply(scalar);
  }

  @Override
  protected String title() {
    return MathematicaFormat.concise("ScalingLayer", scalar);
  }

  @Override
  public Tensor _back() {
    throw new UnsupportedOperationException();
  }
}
