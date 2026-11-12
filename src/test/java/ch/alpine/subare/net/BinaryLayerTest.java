// code by jph
package ch.alpine.subare.net;

import java.util.concurrent.ThreadLocalRandom;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.VectorQ;

class BinaryLayerTest {
  @Test
  void test() {
    NetLayer netLayer = NetChains.binary(NetChains.linear(() -> ElementwiseLayer.gelu(), 2, 5, 1));
    netLayer.initialize(ThreadLocalRandom.current());
    Tensor tensor = netLayer.forward(Tensors.vector(2, 3));
    VectorQ.requireLength(tensor, 1);
  }
}
