// code by jph
package ch.alpine.subare.net;

import java.util.random.RandomGenerator;
import java.util.stream.Stream;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public interface NetLayer {
  default void initialize(RandomGenerator randomGenerator) {
    // TODO ponder about design
  }

  default Stream<Scalar> parameters() {
    return Stream.empty();
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

  /** Update parameters (no-op for layers without params) */
  void update();

  Tensor error(Tensor y);
}
