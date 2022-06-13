// code by jph
package ch.alpine.subare.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.num.Polynomial;

class DiscountFunctionTest {
  @Test
  void testSimple() {
    Tensor coeffs = Tensors.vector(3, 2, -3, 2, .3);
    DiscountFunction discountFunction = DiscountFunction.of(RealScalar.ONE);
    Scalar gain1 = discountFunction.apply(coeffs);
    Scalar gain2 = Polynomial.of(coeffs).apply(RealScalar.ONE);
    assertEquals(gain1, gain2);
  }

  @Test
  void testHorner() {
    Tensor coeffs = Tensors.vector(3, 2, -3, 2, .3);
    Scalar alpha = RealScalar.of(.2);
    DiscountFunction discountFunction = DiscountFunction.of(alpha);
    Scalar gain1 = discountFunction.apply(coeffs);
    Scalar gain2 = Polynomial.of(coeffs).apply(alpha);
    assertEquals(gain1, gain2);
  }

  @Test
  void testFail() {
    assertThrows(Exception.class, () -> DiscountFunction.of(RealScalar.of(1.1)));
  }
}
