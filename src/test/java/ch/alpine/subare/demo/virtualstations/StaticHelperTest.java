// code by jph
package ch.alpine.subare.demo.virtualstations;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.lang.reflect.Modifier;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;

class StaticHelperTest {
  @Test
  void testSimple() {
    Tensor prefix = Tensors.empty();
    Tensor result = StaticHelper.binaryVectors(3, prefix);
    assertEquals(result.length(), 8);
  }

  @Test
  void testBaseCase() {
    Tensor prefix = Tensors.empty();
    Tensor result = StaticHelper.binaryVectors(0, prefix);
    assertEquals(result.length(), 0);
  }

  @Test
  void testMore() {
    Tensor result = StaticHelper.binaryVectors(2, Tensors.vector(1, 2, 3, 4).map(Tensors::of));
    assertEquals(result, Tensors.fromString( //
        "{{1, 1, 1}, {1, 1, 0}, {1, 0, 1}, {1, 0, 0}, {2, 1, 1}, {2, 1, 0}, {2, 0, 1}, {2, 0, 0}, {3, 1, 1}, {3, 1, 0}, {3, 0, 1}, {3, 0, 0}, {4, 1, 1}, {4, 1, 0}, {4, 0, 1}, {4, 0, 0}}"));
  }

  @Test
  void testZeroVectorsEmpty() {
    Tensor prefix = Tensors.empty();
    Tensor result1 = StaticHelper.zeroVectors(1, prefix);
    Tensor result2 = StaticHelper.zeroVectors(10, prefix);
    assertEquals(result1, Tensors.of(Array.zeros(1)));
    assertEquals(result2, Tensors.of(Array.zeros(10)));
  }

  @Test
  void testZeroVectorsFilled() {
    Tensor prefix = Tensors.of(Tensors.vector(1, 2));
    Tensor result0 = StaticHelper.zeroVectors(0, prefix);
    Tensor result1 = StaticHelper.zeroVectors(1, prefix);
    Tensor result2 = StaticHelper.zeroVectors(5, prefix);
    assertEquals(result0, Tensors.of(Tensors.vector(1, 2)));
    assertEquals(result1, Tensors.of(Tensors.vector(1, 2, 0)));
    assertEquals(result2, Tensors.of(Tensors.vector(1, 2, 0, 0, 0, 0, 0)));
    // ---
    prefix.append(Tensors.vector(3, 4));
    Tensor result3 = StaticHelper.zeroVectors(1, prefix);
    Tensor result4 = StaticHelper.zeroVectors(5, prefix);
    assertEquals(result3, Tensors.of(Tensors.vector(1, 2, 0), Tensors.vector(3, 4, 0)));
    assertEquals(result4, Tensors.of(Tensors.vector(1, 2, 0, 0, 0, 0, 0), Tensors.vector(3, 4, 0, 0, 0, 0, 0)));
  }

  @Test
  void testPackageVisibility() {
    assertFalse(Modifier.isPublic(StaticHelper.class.getModifiers()));
  }
}
