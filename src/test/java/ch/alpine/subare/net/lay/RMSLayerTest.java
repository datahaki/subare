// code by jph
package ch.alpine.subare.net.lay;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.mat.Tolerance;

class RMSLayerTest {
  @Test
  void test() {
    NetLayer netLayer = new RMSLayer();
    Tensor y = netLayer.forward(Tensors.vector(2, 3, 4));
    Tolerance.CHOP.requireClose(y, Tensors.vector(0.6432674876302216, 0.9649012314453322, 1.2865349752604431));
  }
}
