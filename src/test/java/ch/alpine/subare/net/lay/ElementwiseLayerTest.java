// code by jph
package ch.alpine.subare.net.lay;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.jet.JetScalar;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;

class ElementwiseLayerTest {
  static void check(NetLayer netLayer) {
    Tensor xx = RandomVariate.of(NormalDistribution.standard(), 2);
    Scalar y_bar = RandomVariate.of(NormalDistribution.standard());
    // ---
    JetScalar y = (JetScalar) netLayer.forward(JetScalar.of(xx));
    Scalar z = (Scalar) netLayer.back(y_bar);
    Scalar r1 = y.vector().Get(1).multiply(y_bar);
    Scalar r2 = z.multiply(xx.Get(1));
    if (r2 instanceof JetScalar js)
      r2 = js.vector().Get(0);
    Tolerance.CHOP.requireClose(r1, r2);
  }
}
