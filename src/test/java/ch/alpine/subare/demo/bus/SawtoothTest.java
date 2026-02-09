// code by jph
package ch.alpine.subare.demo.bus;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Range;

class SawtoothTest {
  @Test
  void testSimple() {
    Sawtooth sawtooth = new Sawtooth(3);
    Tensor s = Range.of(0, 12).maps(sawtooth);
    assertEquals(s, Tensors.vector(0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1));
  }
}
