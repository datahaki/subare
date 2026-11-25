// code by jph
package ch.alpine.subare.net.lay;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.nrm.SoftmaxVector;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/SoftmaxLayer.html">SoftmaxLayer</a> */
public enum SoftmaxLayer implements NetLayer {
  INSTANCE;

  @Override
  public Tensor forward(Tensor logits) {
    return SoftmaxVector.of(logits);
  }

  @Override
  public Tensor forwardBatch(Tensor logits) {
    return Tensor.of(logits.stream().map(SoftmaxVector::of));
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return gradOutput;
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    return gradOutput;
  }

  @Override
  public String toString() {
    return "SoftmaxLayer";
  }
}
