// code by jph
package ch.alpine.subare.core.util.gfx;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.RealScalar;

class UnitClipTest {
  @Test
  void testSimple() {
    assertEquals(UnitClip.FUNCTION.apply(DoubleScalar.INDETERMINATE).toString(), DoubleScalar.INDETERMINATE.toString());
    assertEquals(UnitClip.FUNCTION.apply(DoubleScalar.POSITIVE_INFINITY), DoubleScalar.POSITIVE_INFINITY);
    assertEquals(UnitClip.FUNCTION.apply(RealScalar.of(3)), RealScalar.ONE);
  }
}
