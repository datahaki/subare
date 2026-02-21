// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

public interface Layer {
  /** Forward pass
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

  default Tensor parameters() {
    return Tensors.empty();
  }
}
