// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.nrm.SoftmaxVector;
import ch.alpine.tensor.red.Entrywise;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/SoftmaxLayer.html">SoftmaxLayer</a> */
public class SoftmaxLayer implements NetLayer {
  Tensor outputCache;

  @Override
  public Tensor forward(Tensor x) {
    return outputCache = SoftmaxVector.of(x);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    Scalar dot = (Scalar) gradOutput.dot(outputCache);
    return Entrywise.mul().apply(outputCache, gradOutput.maps(s -> s.subtract(dot)));
  }

  @Override
  public Tensor error(Tensor y) {
    return y.subtract(outputCache); // vector of length d_out
  }
}
