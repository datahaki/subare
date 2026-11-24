// code by jph
package ch.alpine.subare.net.run;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.NetChains;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetModel;
import ch.alpine.subare.net.NetModels;
import ch.alpine.subare.net.lay.ElementwiseLayers;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.mat.Tolerance;

class NetInitializeTest {
  @Test
  void test() {
    Tensor x = Tensors.vector(1, 2);
    NetModel netModel1 = NetModels.softmax_argMax(NetChains.linear(ElementwiseLayers.GELU, 2, 4, 5, 3));
    NetLayer netLayer1 = netModel1.netLayer();
    NetLayers.initialize(netLayer1, new Random(3));
    Tensor y1 = netLayer1.forward(x);
    NetModel netModel2 = NetModels.softmax_argMax(NetChains.linear(ElementwiseLayers.GELU, 2, 4, 5, 3));
    NetLayer netLayer2 = netModel2.netLayer();
    NetLayers.initialize(netLayer2, new Random(3));
    Tensor y2 = netLayer2.forward(x);
    Tolerance.CHOP.requireClose(y1, y2);
    Tensor l1 = Tensor.of(NetLayers.parameters(netLayer1));
    Tensor l2 = Tensor.of(NetLayers.parameters(netLayer2));
    assertEquals(l1, l2);
  }
}
