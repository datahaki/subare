// code by jph
package ch.alpine.subare.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.UnitVector;
import ch.alpine.tensor.mat.IdentityMatrix;

public class RandomChoiceTest {
  @Test
  public void testSimple() {
    Set<Integer> set = new HashSet<>();
    for (int index = 0; index < 100; ++index) {
      int value = RandomChoice.of(Arrays.asList(1, 2, 3, 4));
      set.add(value);
    }
    assertEquals(set.size(), 4);
  }

  @Test
  public void testTensor() {
    Scalar scalar = RandomChoice.of(Tensors.vector(2, 5));
    assertTrue(scalar.equals(RealScalar.of(2)) || scalar.equals(RealScalar.of(5)));
  }

  @Test
  public void testIdentityMatrix() {
    Tensor tensor = RandomChoice.of(IdentityMatrix.of(3));
    assertTrue( //
        tensor.equals(UnitVector.of(3, 0)) || //
            tensor.equals(UnitVector.of(3, 1)) || //
            tensor.equals(UnitVector.of(3, 2)));
  }

  @Test
  public void testTensorFail() {
    AssertFail.of(() -> RandomChoice.of(Tensors.vector()));
    AssertFail.of(() -> RandomChoice.of(RealScalar.ONE));
  }

  @Test
  public void testListEmptyFail() {
    AssertFail.of(() -> RandomChoice.of(Collections.emptyList()));
  }
}
