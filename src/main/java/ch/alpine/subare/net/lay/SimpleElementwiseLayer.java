// code by jph
package ch.alpine.subare.net.lay;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.red.Entrywise;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/ElementwiseLayer.html">ElementwiseLayer</a> */
abstract class SimpleElementwiseLayer extends ElementwiseLayer {
  // TODO Leaky ReLU like relu but with alpha * x for negative
  /** output cache */
  transient Tensor inputCache;
  transient Tensor outputCache;

  public abstract Scalar func(Scalar x);

  @Override
  public final Tensor forward(Tensor x) {
    return outputCache = ((inputCache = x).maps(this::func));
  }

  public abstract Tensor _back();

  @Override
  public Tensor back(Tensor gradOutput) {
    return Entrywise.mul().apply(gradOutput, _back());
  }
}
