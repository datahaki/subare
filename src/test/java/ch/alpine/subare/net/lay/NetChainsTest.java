// code by jph
package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.NetChains;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.run.NetTrainableQ;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.ext.Serialization;

class NetChainsTest {
  @Test
  void test() {
    NetLayer netLayer = NetChains.linear(ElementwiseLayers.GELU, 2, 13, 7, 3);
    Serialization.copy(netLayer);
    NetLayers.initialize(netLayer);
    Tensor tensor = netLayer.forward(Tensors.vector(2, 3));
    assertEquals(tensor.length(), 3);
  }

  @Test
  void testSer() {
    NetLayer netLayer = NetChains.linear(ElementwiseLayers.GELU, 2, 13, 7, 3);
    NetLayer nl2 = NetChains.argMax(netLayer);
    Serialization.copy(nl2);
  }

  @Test
  void testDyn() {
    int[] dims = new int[] { 2, 3, 4, 2, 3, 1 };
    NetLayer netLayer = NetChains.linear(ElementwiseLayers.GELU, dims);
    NetLayers.initialize(netLayer);
    Tensor tensor = netLayer.forward(Tensors.vector(2, 3));
    assertEquals(tensor.length(), 1);
  }

  @Test
  void testTrainable() {
    int[] dims = new int[] { 1, 7, 1 };
    NetLayer netLayer = NetChains.linear(ElementwiseLayers.GELU, dims);
    NetTrainableQ.check(netLayer, dims[0]);
  }

  @Test
  void testSome() {
    NetLayer netLayer = NetChains.argMax(NetChains.linear(ElementwiseLayers.GELU, 2, 7, 3));
    NetLayers.initialize(netLayer);
    Serialization.copy(netLayer);
    NetLayers.trainableLayers(netLayer).forEach(IO::println);
  }
}
