// code by jph
package ch.alpine.subare.net;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.jet.JetScalar;

class ElementwiseLayerTest {
  @Test
  void test() {
    Layer layer = ElementwiseLayer.logSig();
    JetScalar js = JetScalar.of(RealScalar.of(1), 2);
    Tensor x = Tensors.of(js);
    IO.println(x);
    Tensor y = layer.forward(x);
    IO.println(y);
    Tensor z = layer.back(y);
    IO.println(z);
  }
}
