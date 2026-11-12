// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.nrm.SoftmaxVector;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/SoftmaxLayer.html">SoftmaxLayer</a> */
public enum SoftmaxLayer implements NetLayer {
  INSTANCE;

  @Override
  public Tensor forward(Tensor x) {
    return VectorQ.of(x) //
        ? SoftmaxVector.of(x)
        : Tensor.of(x.stream().map(SoftmaxVector::of));
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return gradOutput;
  }
}
