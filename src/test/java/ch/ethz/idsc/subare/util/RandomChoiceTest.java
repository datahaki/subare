// code by jph
package ch.ethz.idsc.subare.util;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.UnitVector;
import ch.alpine.tensor.mat.IdentityMatrix;
import junit.framework.TestCase;

public class RandomChoiceTest extends TestCase {
  public void testSimple() {
    Set<Integer> set = new HashSet<>();
    for (int index = 0; index < 100; ++index) {
      int value = RandomChoice.of(Arrays.asList(1, 2, 3, 4));
      set.add(value);
    }
    assertEquals(set.size(), 4);
  }

  public void testTensor() {
    Scalar scalar = RandomChoice.of(Tensors.vector(2, 5));
    assertTrue(scalar.equals(RealScalar.of(2)) || scalar.equals(RealScalar.of(5)));
  }

  public void testIdentityMatrix() {
    Tensor tensor = RandomChoice.of(IdentityMatrix.of(3));
    assertTrue( //
        tensor.equals(UnitVector.of(3, 0)) || //
            tensor.equals(UnitVector.of(3, 1)) || //
            tensor.equals(UnitVector.of(3, 2)));
  }

  public void testTensorFail() {
    AssertFail.of(() -> RandomChoice.of(Tensors.vector()));
    AssertFail.of(() -> RandomChoice.of(RealScalar.ONE));
  }

  public void testListEmptyFail() {
    AssertFail.of(() -> RandomChoice.of(Collections.emptyList()));
  }
}
