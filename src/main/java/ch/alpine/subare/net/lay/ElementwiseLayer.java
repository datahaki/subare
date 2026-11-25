// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.tensor.Tensor;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/ElementwiseLayer.html">ElementwiseLayer</a> */
public abstract class ElementwiseLayer implements NetLayer, Serializable {
  /** output cache */
  protected transient Tensor inputCache;
  protected transient Tensor outputCache;

  @Override
  public final Tensor forwardBatch(Tensor X) {
    outputCache = forward(X);
    inputCache = X;
    return outputCache;
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
