// code by jph
package ch.alpine.subare.net.lay;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.num.Pi;

class ElementwiseLayersTest {
  @ParameterizedTest
  @EnumSource
  void testScalar(ElementwiseLayers elementwiseLayers) {
    NetLayer netLayer = elementwiseLayers.get();
    Tensor y = netLayer.forward(Pi.VALUE);
    netLayer.back(y);
  }

  @ParameterizedTest
  @EnumSource
  void testJet(ElementwiseLayers elementwiseLayers) {
    NetLayer netLayer = elementwiseLayers.get();
    ElementwiseLayerTest.check(netLayer);
    ElementwiseLayerTest.batch(netLayer);
  }
}
