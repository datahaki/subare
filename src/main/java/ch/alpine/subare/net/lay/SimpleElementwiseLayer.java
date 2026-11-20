// code by jph
package ch.alpine.subare.net.lay;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.red.Entrywise;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/ElementwiseLayer.html">ElementwiseLayer</a> */
public abstract class SimpleElementwiseLayer extends ElementwiseLayer {
  public abstract Scalar func(Scalar x);

  @Override
  public final Tensor forward(Tensor x) {
    return outputCache = ((inputCache = x).maps(this::func));
  }

  /** @return */
  protected abstract Tensor back();

  @Override
  public Tensor back(Tensor gradOutput) {
    return Entrywise.mul().apply(gradOutput, back());
  }
}
