// code by jph
package ch.alpine.subare.net.lay;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.sca.tri.Cos;
import ch.alpine.tensor.sca.tri.Sin;

public class SinLayer extends SimpleElementwiseLayer {
  public static NetLayer of(Scalar omega) {
    return new SinLayer(omega);
  }

  // ---
  private final Scalar omega;

  private SinLayer(Scalar omega) {
    this.omega = omega;
  }

  @Override
  public Scalar func(Scalar x) {
    return Sin.FUNCTION.apply(x.multiply(omega));
  }

  @Override
  protected Tensor back() {
    return inputCache.maps(x -> Cos.FUNCTION.apply(x.multiply(omega)).multiply(omega));
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("SinLayer", omega);
  }
}
