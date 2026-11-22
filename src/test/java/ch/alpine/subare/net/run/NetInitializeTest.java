// code by jph
package ch.alpine.subare.net.run;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
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
    NetLayers.initialize(netLayer1, new Random(3));
    Tensor y1 = netLayer1.forward(x);
    NetLayer netLayer2 = NetChains.argMax(NetChains.linear(ElementwiseLayers.GELU, 2, 4, 5, 3));
    NetLayers.initialize(netLayer2,new Random(3));
    Tensor y2 = netLayer2.forward(x);
    Tolerance.CHOP.requireClose(y1, y2);
    
    Tensor l1 = Tensor.of(NetLayers.parameters(netLayer1));
    Tensor l2 = Tensor.of(NetLayers.parameters(netLayer2));
    assertEquals(l1, l2);
  }
}
