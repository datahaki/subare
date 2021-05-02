// code by jph
package ch.alpine.subare.core.util;

import ch.alpine.subare.core.VsInterface;
import ch.alpine.subare.core.adapter.CosineBasis;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.Clips;
import junit.framework.TestCase;

public class LinearApproximationVsTest extends TestCase {
  public void testSimple() {
    TensorUnaryOperator represent = CosineBasis.create(5, Clips.positive(20));
    VsInterface vs = LinearApproximationVs.create(represent, Tensors.vector(0, 1, 0, 0, 0));
    Scalar value = vs.value(RealScalar.of(10));
    Chop._13.requireAllZero(value);
  }
}
