// code by jph
package ch.alpine.subare.book.ch06.cliff;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensors;

class CliffwalkTest {
  @Test
  void testStates() {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    assertEquals(cliffwalk.states().length(), 12 * 3 + 2);
  }

  @Test
  void testMove() {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    assertEquals(cliffwalk.move(Tensors.vector(0, 0), Tensors.vector(0, -1)), Tensors.vector(0, 0));
    assertEquals(cliffwalk.move(Tensors.vector(0, 0), Tensors.vector(0, 1)), Tensors.vector(0, 1));
    assertEquals(cliffwalk.move(Tensors.vector(0, 3), Tensors.vector(0, 1)), Tensors.vector(0, 3));
    assertEquals(cliffwalk.move(Tensors.vector(0, 3), Tensors.vector(0, -1)), Tensors.vector(0, 2));
    assertEquals(cliffwalk.move(Tensors.vector(11, 3), Tensors.vector(0, -1)), Tensors.vector(11, 3));
  }

  @Test
  void testCliff() {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    assertFalse(cliffwalk.isCliff(Tensors.vector(0, 3)));
    assertTrue(cliffwalk.isCliff(Tensors.vector(1, 3)));
    assertTrue(cliffwalk.isCliff(Tensors.vector(10, 3)));
    assertFalse(cliffwalk.isCliff(Tensors.vector(11, 3)));
  }

  @Test
  void testReward() {
    Cliffwalk cliffwalk = new Cliffwalk(12, 4);
    assertEquals(cliffwalk.reward(Tensors.vector(11, 2), null, Tensors.vector(11, 3)), RealScalar.ONE);
    assertEquals(cliffwalk.reward(Tensors.vector(11, 3), null, Tensors.vector(11, 3)), RealScalar.ZERO);
    // assertEquals(cliffwalk.move(, Tensors.vector(0, -1)), Tensors.vector(0, 0));
    // assertEquals(cliffwalk.move(Tensors.vector(0, 0), Tensors.vector(0, 1)), Tensors.vector(0, 1));
    // assertEquals(cliffwalk.move(Tensors.vector(0, 3), Tensors.vector(0, 1)), Tensors.vector(0, 3));
    // assertEquals(cliffwalk.move(Tensors.vector(0, 3), Tensors.vector(0, -1)), Tensors.vector(0, 2));
    // assertEquals(cliffwalk.move(Tensors.vector(11, 3), Tensors.vector(0, -1)), Tensors.vector(11, 3));
  }
}
