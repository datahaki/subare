// code by jph
package ch.alpine.subare.net.lay;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.tensor.alg.Array;

class DropoutLayerTest {
  @Test
  void test() {
    NetLayer dropoutLayer = DropoutLayer.of(0.3);
    dropoutLayer.forward(Array.zeros(10));
  }
}
