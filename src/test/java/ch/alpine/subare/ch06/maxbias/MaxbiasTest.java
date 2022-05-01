// code by jph
package ch.alpine.subare.ch06.maxbias;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensors;

class MaxbiasTest {
  @Test
  public void testMove() {
    Maxbias maxbias = new Maxbias(3);
    assertEquals(maxbias.move(RealScalar.ONE, RealScalar.ONE), RealScalar.ZERO);
    assertEquals(maxbias.move(RealScalar.of(2), RealScalar.of(1)), RealScalar.of(3));
    assertEquals(maxbias.move(RealScalar.of(2), RealScalar.of(-1)), RealScalar.of(1));
  }

  @Test
  public void testTerminal() {
    Maxbias maxbias = new Maxbias(3);
    assertTrue(maxbias.isTerminal(RealScalar.ZERO));
    assertFalse(maxbias.isTerminal(RealScalar.ONE));
    assertFalse(maxbias.isTerminal(RealScalar.of(2)));
    assertTrue(maxbias.isTerminal(RealScalar.of(3)));
  }

  @Test
  public void testReward() {
    Maxbias maxbias = new Maxbias(3);
    assertEquals(maxbias.reward(RealScalar.of(2), RealScalar.of(1), RealScalar.of(3)), RealScalar.ZERO);
  }

  @Test
  public void testStarting() {
    Maxbias maxbias = new Maxbias(3);
    assertEquals(maxbias.startStates(), Tensors.vector(2));
  }
}
