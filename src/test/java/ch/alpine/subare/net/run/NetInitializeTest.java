// code by jph
package ch.alpine.subare.net.run;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.NetExtract;
import ch.alpine.subare.net.NetInitialize;
import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.subare.net.lay.ElementwiseLayer;
import ch.alpine.subare.net.lay.NetChains;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.mat.Tolerance;

class NetInitializeTest {
  @Test
  void test() {
    Tensor x = Tensors.vector(1, 2);
    NetLayer netLayer1 = NetChains.argMax(NetChains.linear(() -> ElementwiseLayer.gelu(), 2, 4, 5, 3));
    NetInitialize.of(netLayer1, new Random(3));
    Tensor y1 = netLayer1.forward(x);
    NetLayer netLayer2 = NetChains.argMax(NetChains.linear(() -> ElementwiseLayer.gelu(), 2, 4, 5, 3));
    NetInitialize.of(netLayer2, new Random(3));
    Tensor y2 = netLayer2.forward(x);
    Tolerance.CHOP.requireClose(y1, y2);
    List<Scalar> l1 = new NetExtract(netLayer1).all().toList();
    List<Scalar> l2 = new NetExtract(netLayer2).all().toList();
    assertEquals(l1, l2);
  }
}
