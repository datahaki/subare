// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.tensor.Tensor;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/ElementwiseLayer.html">ElementwiseLayer</a> */
public abstract class ElementwiseLayer implements NetLayer, Serializable {
  /** output cache */
  protected transient Tensor inputCache;
  protected transient Tensor outputCache;

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
    return title();
  }
}
