// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.api.TensorUnaryOperator;

public enum NetLayers {
  ;
  public static ScalarUnaryOperator asSUO11(NetLayer netLayer) {
    return s -> netLayer.forward(Tensors.of(s)).Get(0);
  }

  public static ScalarUnaryOperator asSUO(NetLayer netLayer) {
    return s -> (Scalar) netLayer.forward(s);
  }

  public static TensorUnaryOperator asTUO(NetLayer netLayer) {
    return netLayer::forward;
  }
}
