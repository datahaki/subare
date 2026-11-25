// code by jph
package ch.alpine.subare.net.lay;

import java.util.stream.Stream;

enum NetExtract {
  ;
  static Stream<NetLayer> flatten(NetLayer netLayer) {
    return Stream.concat( //
        Stream.of(netLayer), //
        netLayer.nested().flatMap(layer -> flatten(layer)));
  }
}
