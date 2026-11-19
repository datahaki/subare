// code by jph
package ch.alpine.subare.net;

import java.util.List;
import java.util.stream.Stream;

import ch.alpine.subare.net.api.NetLayer;

public enum NetExtract {
  ;
  public static Stream<NetLayer> flatten(NetLayer netLayer) {
    List<NetLayer> list = netLayer.nested().toList();
    if (list.isEmpty())
      return Stream.of(netLayer);
    return netLayer.nested().flatMap(l -> flatten(l));
  }
}
