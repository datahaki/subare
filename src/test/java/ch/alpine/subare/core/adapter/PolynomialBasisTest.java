// code by jph
package ch.alpine.subare.core.adapter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.sca.Clips;

class PolynomialBasisTest {
  @Test
  public void testLo() {
    TensorUnaryOperator tuo = new PolynomialBasis(4, Clips.interval(50, 100));
    assertEquals(tuo.apply(RealScalar.of(50)), Tensors.vector(1, 0, 0, 0));
    assertEquals(tuo.apply(RealScalar.of(100)), Tensors.vector(1, 1, 1, 1));
    assertEquals(tuo.apply(RealScalar.of(75)), Tensors.fromString("{1, 1/2, 1/4, 1/8}"));
  }

  @Test
  public void testFail() {
    TensorUnaryOperator tuo = new PolynomialBasis(4, Clips.interval(50, 100));
    assertThrows(Exception.class, () -> tuo.apply(RealScalar.ZERO));
  }
}
