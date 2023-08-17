// code by jph
package ch.alpine.subare.math;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.num.Pi;

class IndexTest {
  @Test
  void testSimple() {
    Tensor tensor = Tensors.vector(5, 7, 11);
    Index index = Index.build(tensor);
    assertEquals(index.of(RealScalar.of(5)), 0);
    assertEquals(index.of(RealScalar.of(7)), 1);
    assertEquals(index.of(RealScalar.of(11)), 2);
    assertThrows(Exception.class, () -> index.of(RealScalar.of(8)));
  }

  @Test
  void testDuplicateFail() {
    Tensor tensor = Tensors.vector(5, 7, 11, 7);
    assertThrows(Exception.class, () -> Index.build(tensor));
  }

  @Test
  void testScalarFail() {
    assertThrows(Exception.class, () -> Index.build(Pi.VALUE));
  }
}
