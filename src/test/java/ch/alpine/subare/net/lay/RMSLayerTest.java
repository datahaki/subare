// code by jph
package ch.alpine.subare.net.lay;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;

class RMSLayerTest {
  @Test
  void test() {
    NetLayer netLayer = new RMSLayer();
    Tensor y = netLayer.forward(Tensors.vector(2, 3, 4));
    Tolerance.CHOP.requireClose(y, Tensors.vector(0.6432674876302216, 0.9649012314453322, 1.2865349752604431));
    netLayer.back(Tensors.vector(0.2, -0.3, 0.1));
  }

  @Test
  void testBatch() {
    NetLayer netLayer = new RMSLayer();
    Tensor X = RandomVariate.of(NormalDistribution.standard(), 10, 3);
    Tensor Y1 = Tensor.of(X.stream().map(netLayer::forward));
    Tensor Y2 = netLayer.forwardBatch(X);
    Tolerance.CHOP.requireClose(Y1, Y2);
    Tensor grad = RandomVariate.of(NormalDistribution.standard(), 10, 3);
    netLayer.backBatch(grad);
  }
}
