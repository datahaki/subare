// code by jph
package ch.alpine.subare.net.lay;

import java.util.stream.Stream;

import ch.alpine.subare.net.err.NetLoss;
import ch.alpine.tensor.Tensor;

public interface NetLayer {
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
  default NetLoss error(Tensor target) {
    throw new UnsupportedOperationException();
  }

  /** @param gradOutput
   * @return */
  Tensor back(Tensor gradOutput);

  default Stream<NetLayer> stream() {
    return Stream.of(this);
  }
}
