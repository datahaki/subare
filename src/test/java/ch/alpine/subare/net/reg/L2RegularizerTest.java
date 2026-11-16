// code by jph
package ch.alpine.subare.net.reg;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;

class L2RegularizerTest {
  @Test
  void test() {
    L2Regularizer l2Regularizer = new L2Regularizer(RealScalar.of(0.25));
    assertEquals(l2Regularizer.toString(), "L2[0.25]");
  }
}
