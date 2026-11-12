// code by jph
package ch.alpine.subare.net;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.ext.Serialization;

class NetChainsTest {
  @Test
  void test() throws ClassNotFoundException, IOException {
    NetLayer netLayer = NetChains.linear(() -> ElementwiseLayer.gelu(), 2, 13, 7, 3);
    Serialization.copy(netLayer);
    netLayer.initialize(ThreadLocalRandom.current());
    Tensor tensor = netLayer.forward(Tensors.vector(2, 3));
    assertEquals(tensor.length(), 3);
  }

  @Test
  void testSer() throws ClassNotFoundException, IOException {
    NetLayer netLayer = NetChains.linear(() -> ElementwiseLayer.gelu(), 2, 13, 7, 3);
    NetLayer nl2 = NetChains.argMax(netLayer);
    Serialization.copy(nl2);
  }

  @Test
  void testDyn() {
    int[] dims = new int[] { 2, 3, 4, 2, 3, 1 };
    NetLayer netLayer = NetChains.linear(() -> ElementwiseLayer.gelu(), dims);
    netLayer.initialize(ThreadLocalRandom.current());
    Tensor tensor = netLayer.forward(Tensors.vector(2, 3));
    assertEquals(tensor.length(), 1);
  }

  @Test
  void testTrainable() {
    int[] dims = new int[] { 1, 7, 1 };
    NetLayer netLayer = NetChains.linear(() -> ElementwiseLayer.gelu(), dims);
    NetTrainableQ.check(netLayer, dims[0]);
  }
}
