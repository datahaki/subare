// code by jph
package ch.alpine.subare.util;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.api.VsInterface;
import ch.alpine.subare.math.CosineBasis;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.Clips;

class LinearApproximationVsTest {
  @Test
  void testSimple() {
    TensorUnaryOperator represent = new CosineBasis(5, Clips.positive(20));
    VsInterface vs = LinearApproximationVs.create(represent, Tensors.vector(0, 1, 0, 0, 0));
    Scalar value = vs.value(RealScalar.of(10));
    Chop._13.requireAllZero(value);
  }
}
