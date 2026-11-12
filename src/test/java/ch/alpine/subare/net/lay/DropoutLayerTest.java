// code by jph
package ch.alpine.subare.net.lay;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.lay.DropoutLayer;
import ch.alpine.tensor.alg.Array;

class DropoutLayerTest {
  @Test
  void test() {
    DropoutLayer dropoutLayer = new DropoutLayer();
    dropoutLayer.forward(Array.zeros(10));
  }
}
