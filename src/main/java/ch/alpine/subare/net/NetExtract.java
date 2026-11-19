// code by jph
package ch.alpine.subare.net;

import java.util.List;
import java.util.stream.Stream;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.alg.Flatten;

public enum NetExtract {
  ;
  public static Stream<NetLayer> flatten(NetLayer netLayer) {
    List<NetLayer> list = netLayer.nested().toList();
    if (list.isEmpty())
      return Stream.of(netLayer);
    return netLayer.nested().flatMap(l -> flatten(l));
  }

  public static Stream<Scalar> parameters(NetLayer netLayer) {
    return flatten(netLayer) //
        .map(NetLayer::parameters) //
        .flatMap(Flatten::scalars);
  }

  public static Scalar reg_loss(NetLayer netLayer, boolean weightDecay) {
    return flatten(netLayer) //
        .map(l -> l.reg_loss(weightDecay)) //
        .reduce(Scalar::add) //
        .orElseThrow();
  }
}
