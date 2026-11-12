// code by jph
package ch.alpine.subare.net;

import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.ScalarUnaryOperator;

public enum NetWrap {
  ;
  public static ScalarUnaryOperator asSUO11(NetLayer netLayer) {
    return s -> netLayer.forward(Tensors.of(s)).Get(0);
  }
}
