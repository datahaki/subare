// code by jph
package ch.ethz.idsc.subare.core.adapter;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.sca.Clips;
import ch.ethz.idsc.subare.util.AssertFail;
import junit.framework.TestCase;

public class CosineBasisTest extends TestCase {
  public void testLo() {
    TensorUnaryOperator fb = CosineBasis.create(4, Clips.interval(50, 100));
    Tensor result = fb.apply(RealScalar.of(50));
    assertEquals(result, Tensors.vector(1, 1, 1, 1));
  }

  public void testHi() {
    TensorUnaryOperator fb = CosineBasis.create(4, Clips.interval(0, 100));
    Tensor result = fb.apply(RealScalar.of(100));
    assertEquals(result, Tensors.vector(1, -1, 1, -1));
  }

  public void testFail() {
    TensorUnaryOperator tuo = CosineBasis.create(4, Clips.interval(50, 100));
    AssertFail.of(() -> tuo.apply(RealScalar.ZERO));
  }
}
