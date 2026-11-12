// code by jph
package ch.alpine.subare.net;

import java.util.stream.Stream;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public interface NetLayer {
  // TODO API not in here
  default Stream<Scalar> parameters() {
    return Stream.empty();
  }

  // TODO API not in here
  default void training() {
  }

  /** forward pass
   * 
   * @param x
   * @return */
  Tensor forward(Tensor x);

  /** ErrorThe element-wise or raw difference between a model's
   * prediction and the ground truth target (e.g., error = y - target ).
   * It is vector-valued or tensor-valued per sample.
   * 
   * @param target
   * @return */
  default NetError error(Tensor target) {
    throw new UnsupportedOperationException();
  }

  /** @param gradOutput
   * @return */
  Tensor back(Tensor gradOutput);

  /** Update parameters (no-op for layers without params) */
  // TODO API not in here
  default void update(Scalar lr) {
  }

  default Stream<NetLayer> stream() {
    return Stream.of(this);
  }
}
