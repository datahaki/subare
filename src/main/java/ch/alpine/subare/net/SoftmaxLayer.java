// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.nrm.SoftmaxVector;
import ch.alpine.tensor.red.Entrywise;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/SoftmaxLayer.html">SoftmaxLayer</a> */
public class SoftmaxLayer implements Layer {
  Tensor lastOutput;

  @Override
  public Tensor forward(Tensor x) {
    return lastOutput = SoftmaxVector.of(x);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    Scalar dot = (Scalar) gradOutput.dot(lastOutput);
    return Entrywise.mul().apply(lastOutput, gradOutput.maps(s -> s.subtract(dot)));
  }

  @Override
  public void update() {
  }

  @Override
  public Tensor error(Tensor y) {
    throw new IllegalStateException();
  }

  @Override
  public Tensor parameters() {
    return Tensors.empty();
  }
}
