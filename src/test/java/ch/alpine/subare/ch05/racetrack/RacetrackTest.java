// code by jph
package ch.alpine.subare.ch05.racetrack;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.util.Index;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.ArrayQ;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.io.Import;

class RacetrackTest {
  @Test
  void testStartAction() {
    Racetrack racetrack = new Racetrack(Import.of("/ch05/track0.png"), 3);
    Index statesIndex = Index.build(racetrack.states());
    assertEquals(statesIndex.size(), 724);
    assertEquals(racetrack.statesStart, Tensors.fromString("{{1, 0, 0, 0}, {2, 0, 0, 0}, {3, 0, 0, 0}}"));
    assertEquals(racetrack.statesTerminal.length() % 3, 1);
    assertThrows(Exception.class, () -> racetrack.actions(Tensors.vector(1, 0, 0)));
  }

  @Test
  void testMove() {
    Racetrack racetrack = new Racetrack(Import.of("/ch05/track0.png"), 3);
    assertEquals(Dimensions.of(racetrack.image()), List.of(8, 11, 4));
    Tensor start = Tensors.vector(1, 0, 0, 0);
    assertTrue(racetrack.isStart(start));
    assertFalse(racetrack.isTerminal(start));
    Tensor next = racetrack.integrate(start, Tensors.vector(1, 1)); // vy
    assertTrue(racetrack.statesIndex.containsKey(next));
    Tensor move = racetrack.move(start, Tensors.vector(2, 3));
    assertEquals(move, Tensors.vector(3, 3, 2, 3));
  }

  @Test
  void testSome() {
    Racetrack racetrack = new Racetrack(Import.of("/ch05/track0.png"), 3);
    for (Tensor state : racetrack.states())
      racetrack.actions(state);
  }

  @Test
  void testArray() {
    Racetrack racetrack = new Racetrack(Import.of("/ch05/track0.png"), 3);
    ArrayQ.require(racetrack.actions);
  }
}
