// code by jph
package ch.alpine.subare.net.lay;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.num.Pi;

class ScalingLayerTest {
  @Test
  void test() {
    ScalingLayer scalingLayer = new ScalingLayer(Pi.VALUE);
    ElementwiseLayerTest.check(scalingLayer);
  }
}
