// code by jph
package ch.alpine.subare.net.run;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.mat.Tolerance;

class NetInitializeTest {
  @Test
  void test() {
    Tensor x = Tensors.vector(1, 2);
    NetLayer netLayer1 = NetChains.argMax(NetChains.linear(ElementwiseLayers.GELU, 2, 4, 5, 3));
    netLayer1.initialize(new Random(3));
    Tensor y1 = netLayer1.forward(x);
    NetLayer netLayer2 = NetChains.argMax(NetChains.linear(ElementwiseLayers.GELU, 2, 4, 5, 3));
    netLayer2.initialize(new Random(3));
    Tensor y2 = netLayer2.forward(x);
    Tolerance.CHOP.requireClose(y1, y2);
    Tensor l1 = netLayer1.parameters();
    Tensor l2 = netLayer2.parameters();
    assertEquals(l1, l2);
  }
}
