// code by jph
package ch.alpine.subare.core.adapter;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.util.CosineBasis;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.sca.Clips;

class CosineBasisTest {
  @Test
  void testLo() {
    TensorUnaryOperator fb = new CosineBasis(4, Clips.interval(50, 100));
    Tensor result = fb.apply(RealScalar.of(50));
    assertEquals(result, Tensors.vector(1, 1, 1, 1));
  }

  @Test
  void testHi() {
    TensorUnaryOperator fb = new CosineBasis(4, Clips.interval(0, 100));
    Tensor result = fb.apply(RealScalar.of(100));
    assertEquals(result, Tensors.vector(1, -1, 1, -1));
  }

  @Test
  void testFail() {
    TensorUnaryOperator tuo = new CosineBasis(4, Clips.interval(50, 100));
    assertThrows(Exception.class, () -> tuo.apply(RealScalar.ZERO));
  }
}
