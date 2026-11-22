// code by jph
package ch.alpine.subare.net;

import java.util.random.RandomGenerator;
import java.util.stream.Stream;

import ch.alpine.tensor.Scalar;
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

  /** visits all members
   * useful for pointwise operations such as
   * {@link #initialize(RandomGenerator)}
   * {@link #parameters()}
   * {@link #gradient(boolean)}
   * {@link #increment(boolean, Tensor, Scalar)}
   * {@link #reg_loss(boolean)}
   * 
   * @return all net layers directly harbored by this instance */
  default Stream<NetLayer> nested() {
    return Stream.empty();
  }
}
