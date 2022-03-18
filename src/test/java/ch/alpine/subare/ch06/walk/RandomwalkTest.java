// code by jph
package ch.alpine.subare.ch06.walk;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Range;

public class RandomwalkTest {
  @Test
  public void testSmall() {
    Randomwalk rw = new Randomwalk(5);
    assertEquals(rw.states(), Range.of(0, 7));
    assertEquals(rw.startStates(), Tensors.vector(3));
    assertTrue(rw.isTerminal(RealScalar.of(6)));
  }

  @Test
  public void testLarge() {
    Randomwalk rw = new Randomwalk(19);
    assertEquals(rw.states(), Range.of(0, 21));
    assertEquals(rw.startStates(), Tensors.vector(10));
    assertTrue(rw.isTerminal(RealScalar.of(20)));
  }
}
