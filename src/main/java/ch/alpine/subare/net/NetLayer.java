// code by jph
package ch.alpine.subare.net;

import java.util.random.RandomGenerator;
import java.util.stream.Stream;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public interface NetLayer {
  default void initialize(RandomGenerator randomGenerator) {
  }

  default Stream<Scalar> parameters() {
    return Stream.empty();
  }

  /** Update parameters (no-op for layers without params) */
  default void update() {
  }

  /** forward pass
   * 
   * @param x
   * @return */
  Tensor forward(Tensor x);

  /** Backward pass
   * gradOutput = dL/da
   * returns dL/dinput
   * 
   * @param gradOutput
   * @return */
  Tensor back(Tensor gradOutput);

  Tensor error(Tensor y);
}
