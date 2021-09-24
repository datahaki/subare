// code by jph
package ch.alpine.subare.core;

import ch.alpine.subare.util.AssertFail;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.num.Polynomial;
import junit.framework.TestCase;

public class DiscountFunctionTest extends TestCase {
  public void testSimple() {
    Tensor coeffs = Tensors.vector(3, 2, -3, 2, .3);
    DiscountFunction discountFunction = DiscountFunction.of(RealScalar.ONE);
    Scalar gain1 = discountFunction.apply(coeffs);
    Scalar gain2 = Polynomial.of(coeffs).apply(RealScalar.ONE);
    assertEquals(gain1, gain2);
  }

  public void testHorner() {
    Tensor coeffs = Tensors.vector(3, 2, -3, 2, .3);
    Scalar alpha = RealScalar.of(.2);
    DiscountFunction discountFunction = DiscountFunction.of(alpha);
    Scalar gain1 = discountFunction.apply(coeffs);
    Scalar gain2 = Polynomial.of(coeffs).apply(alpha);
    assertEquals(gain1, gain2);
  }

  public void testFail() {
    AssertFail.of(() -> DiscountFunction.of(RealScalar.of(1.1)));
  }
}
