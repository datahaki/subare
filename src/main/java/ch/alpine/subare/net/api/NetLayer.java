// code by jph
package ch.alpine.subare.net.api;

import java.util.stream.Stream;

import ch.alpine.tensor.Tensor;

public interface NetLayer {
  /** @param x
   * @return forward of x */
  Tensor forward(Tensor x);

  /** @param X
   * @return forward of entire batch */
  Tensor forwardBatch(Tensor X);

  /** ErrorThe element-wise or raw difference between a model's
   * prediction and the ground truth target (e.g., error = y - target ).
   * It is vector-valued or tensor-valued per sample.
   * 
   * @param target
   * @return */
  default NetLoss error(Tensor target) {
    throw new UnsupportedOperationException();
  }

  /** @param gradOutput
   * @return */
  Tensor back(Tensor gradOutput);

  /** back propagation of gradient of entire batch
   * 
   * @param gradOutput
   * @return */
  Tensor backBatch(Tensor gradOutput);

  /** @return stream of layers nested inside this net layer */
  default Stream<NetLayer> stream() {
    return Stream.of(this);
  }
}
