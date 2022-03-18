// code by jph
package ch.alpine.subare.util;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.num.Pi;

public class IndexTest {
  @Test
  public void testSimple() {
    Tensor tensor = Tensors.vector(5, 7, 11);
    Index index = Index.build(tensor);
    assertEquals(index.of(RealScalar.of(7)), 1);
    AssertFail.of(() -> index.of(RealScalar.of(8)));
  }

  @Test
  public void testScalarFail() {
    AssertFail.of(() -> Index.build(Pi.VALUE));
  }
}
