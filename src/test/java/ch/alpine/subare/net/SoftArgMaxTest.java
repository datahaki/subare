// code by jph
package ch.alpine.subare.net;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.pdf.d.BernoulliDistribution;

class SoftArgMaxTest {
  @Test
  void testVector() {
    NetLayer netLayer1 = new SoftArgMax();
    NetLayer netLayer2 = NetChain.of(SoftmaxLayer.INSTANCE, new ArgMaxLayer());
    Tensor x = RandomVariate.of(NormalDistribution.standard(), 6);
    Tensor y1 = netLayer1.forward(x);
    Tensor y2 = netLayer2.forward(x);
    Tolerance.CHOP.requireClose(y1, y2);
    Tensor target = RealScalar.ONE;
    NetError e1 = netLayer1.error(target);
    NetError e2 = netLayer2.error(target);
    Scalar l1 = e1.loss();
    Scalar l2 = e2.loss();
    Tolerance.CHOP.requireClose(l1, l2);
    Tolerance.CHOP.requireClose(e1.gradOutput(), e2.gradOutput());
    netLayer1.back(e1.gradOutput());
    netLayer2.back(e2.gradOutput());
  }

  @Test
  void testmatrix() {
    int batch_size = 10;
    NetLayer netLayer1 = new SoftArgMax();
    NetLayer netLayer2 = NetChain.of(SoftmaxLayer.INSTANCE, new ArgMaxLayer());
    Tensor x = RandomVariate.of(NormalDistribution.standard(), batch_size, 4);
    Tensor y1 = netLayer1.forward(x);
    Tensor y2 = netLayer2.forward(x);
    Tolerance.CHOP.requireClose(y1, y2);
    Tensor target = RandomVariate.of(BernoulliDistribution.of(0.3), batch_size);
    NetError e1 = netLayer1.error(target);
    NetError e2 = netLayer2.error(target);
    Scalar l1 = e1.loss();
    Scalar l2 = e2.loss();
    Tolerance.CHOP.requireClose(l1, l2);
    Tolerance.CHOP.requireClose(e1.gradOutput(), e2.gradOutput());
    netLayer1.back(e1.gradOutput());
    netLayer2.back(e2.gradOutput());
  }
}
