// code by jph
package ch.alpine.subare.net.lay;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.sca.Sign;

public class LeakyReLU extends SimpleElementwiseLayer {
  public static NetLayer of(Scalar alpha) {
    return new LeakyReLU(alpha);
  }

  // ---
  private final Scalar alpha;

  private LeakyReLU(Scalar alpha) {
    this.alpha = alpha;
  }

  @Override
  public Scalar func(Scalar x) {
    return Sign.isPositive(x) //
        ? x
        : alpha.multiply(x);
  }

  @Override
  protected Tensor back() {
    return inputCache.maps(x -> Sign.isPositive(x) //
        ? alpha.one()
        : alpha);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("LeakyReLU", alpha);
  }
}
