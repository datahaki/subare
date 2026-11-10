// code by jph
package ch.alpine.subare.net;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.concurrent.ThreadLocalRandom;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

class NetChainsTest {
  @Test
  void test() {
    NetLayer netLayer = NetChains.sequence(() -> ElementwiseLayer.gelu(), 2, 13, 7, 3);
    netLayer.initialize(ThreadLocalRandom.current());
    Tensor tensor = netLayer.forward(Tensors.vector(2, 3));
    assertEquals(tensor.length(), 3);
  }

  @Test
  void testDyn() {
    int[] dims = new int[] { 2, 3, 4, 2, 3, 1 };
    NetLayer netLayer = NetChains.sequence(() -> ElementwiseLayer.gelu(), dims);
    netLayer.initialize(ThreadLocalRandom.current());
    Tensor tensor = netLayer.forward(Tensors.vector(2, 3));
    assertEquals(tensor.length(), 1);
  }
}
