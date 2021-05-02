// code by jph
package ch.ethz.idsc.subare.core.adapter;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.sca.Clips;
import ch.ethz.idsc.subare.util.AssertFail;
import junit.framework.TestCase;

public class PolynomialBasisTest extends TestCase {
  public void testLo() {
    TensorUnaryOperator tuo = PolynomialBasis.create(4, Clips.interval(50, 100));
    assertEquals(tuo.apply(RealScalar.of(50)), Tensors.vector(1, 0, 0, 0));
    assertEquals(tuo.apply(RealScalar.of(100)), Tensors.vector(1, 1, 1, 1));
    assertEquals(tuo.apply(RealScalar.of(75)), Tensors.fromString("{1, 1/2, 1/4, 1/8}"));
  }

  public void testFail() {
    TensorUnaryOperator tuo = PolynomialBasis.create(4, Clips.interval(50, 100));
    AssertFail.of(() -> tuo.apply(RealScalar.ZERO));
  }
}
