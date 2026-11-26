// code by jph
package ch.alpine.subare.net.lay;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.io.MathematicaFormat;

public class ScalingLayer extends ElementwiseLayer {
  private final Scalar scalar;

  public ScalingLayer(Scalar scalar) {
    this.scalar = scalar;
  }

  @Override
  public Tensor forward(Tensor x) {
    return x.multiply(scalar);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return gradOutput.multiply(scalar);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("ScalingLayer", scalar);
  }
}
