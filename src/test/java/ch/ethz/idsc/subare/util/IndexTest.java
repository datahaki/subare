// code by jph
package ch.ethz.idsc.subare.util;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.num.Pi;
import junit.framework.TestCase;

public class IndexTest extends TestCase {
  public void testSimple() {
    Tensor tensor = Tensors.vector(5, 7, 11);
    Index index = Index.build(tensor);
    assertEquals(index.of(RealScalar.of(7)), 1);
    AssertFail.of(() -> index.of(RealScalar.of(8)));
  }

  public void testScalarFail() {
    AssertFail.of(() -> Index.build(Pi.VALUE));
  }
}
