// code by jph
package ch.alpine.subare.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.ext.Serialization;

class FairArgTest {
  @Test
  void testMaxIsFair() throws ClassNotFoundException, IOException {
    Tensor d = Tensors.vectorDouble(3, .3, 3, .6, 3);
    Set<Integer> set = new HashSet<>();
    FairArg fairArg = Serialization.copy(FairArg.max(d));
    for (int index = 0; index < 100; ++index)
      set.add(fairArg.nextRandomIndex());
    assertEquals(set.size(), 3);
  }

  @Test
  void testMinIsFair() {
    Tensor d = Tensors.vectorDouble(3, .3, 3, .6, 3, .3);
    Set<Integer> set = new HashSet<>();
    FairArg fairArg = FairArg.min(d);
    for (int index = 0; index < 100; ++index)
      set.add(fairArg.nextRandomIndex());
    assertEquals(set.size(), 2);
  }

  @Test
  void testInfty() {
    Tensor d = Tensors.of( //
        DoubleScalar.POSITIVE_INFINITY, RealScalar.ONE, //
        DoubleScalar.POSITIVE_INFINITY, DoubleScalar.POSITIVE_INFINITY);
    FairArg fairArg = FairArg.max(d);
    assertEquals(fairArg.optionsCount(), 3);
    List<Integer> list = fairArg.options();
    assertEquals(list, Arrays.asList(0, 2, 3));
  }

  @Test
  void testEmptyFail() {
    assertThrows(Exception.class, () -> FairArg.max(Tensors.empty()));
  }

  @Test
  void testScalarFail() {
    assertThrows(Exception.class, () -> FairArg.max(RealScalar.ONE));
  }
}
