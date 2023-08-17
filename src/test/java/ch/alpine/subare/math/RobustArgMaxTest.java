// code by jph
package ch.alpine.subare.math;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.mat.HilbertMatrix;
import ch.alpine.tensor.sca.Chop;

class RobustArgMaxTest {
  @Test
  void testSimple() {
    Tensor tensor = Tensors.vector(-9, 0, 0.9999999, .3, 1, 0.9999999);
    RobustArgMax robustArgMax = new RobustArgMax(Chop._06);
    int index = robustArgMax.of(tensor);
    assertEquals(index, 2);
  }

  @Test
  void testOptions() {
    RobustArgMax robustArgMax = new RobustArgMax(Chop._04);
    IntStream options = robustArgMax.options(Tensors.vector(0, 0.99, 1, 1.00001, -3, 0.999999));
    List<Integer> list = options.boxed().collect(Collectors.toList());
    assertEquals(list, Arrays.asList(2, 3, 5));
  }

  @Test
  void testFailEmpty() {
    RobustArgMax robustArgMax = new RobustArgMax(Chop._06);
    try {
      robustArgMax.options(Tensors.empty());
      assertFalse(false);
    } catch (Exception exception) {
      // ---
    }
  }

  @Test
  void testFailMatrix() {
    RobustArgMax robustArgMax = new RobustArgMax(Chop._06);
    try {
      robustArgMax.options(HilbertMatrix.of(3, 4));
      assertFalse(false);
    } catch (Exception exception) {
      // ---
    }
    try {
      robustArgMax.of(HilbertMatrix.of(3, 4));
      assertFalse(false);
    } catch (Exception exception) {
      // ---
    }
  }
}
