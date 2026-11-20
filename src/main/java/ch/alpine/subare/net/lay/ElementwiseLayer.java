// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.Objects;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.io.MathematicaFormat;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/ElementwiseLayer.html">ElementwiseLayer</a> */
abstract class ElementwiseLayer implements NetLayer, Serializable {
  /** output cache */
  transient Tensor inputCache;
  transient Tensor outputCache;

  @Override
  public final Tensor forwardBatch(Tensor X) {
    return outputCache = forward(inputCache = X);
  }

  @Override
  public final Tensor backBatch(Tensor gradOutput) {
    return back(gradOutput);
  }

  protected abstract String title();

  @Override
  public final String toString() {
    return MathematicaFormat.concise(title(), Objects.isNull(inputCache) ? "" : Dimensions.of(inputCache));
  }
}
