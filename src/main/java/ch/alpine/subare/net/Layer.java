// code by jph
package ch.alpine.subare.net;

import java.util.function.BiFunction;

import ch.alpine.tensor.Tensor;

public interface Layer {
  public static BiFunction<Tensor, Layer, Tensor> back() {
    return (d, layer) -> layer.back(d);
  }

  public static BiFunction<Tensor, Layer, Tensor> forward() {
    return (x, layer) -> layer.forward(x);
  }

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

  Tensor parameters();
}
