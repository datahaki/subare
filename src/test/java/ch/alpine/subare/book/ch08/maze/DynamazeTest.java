// code by jph
package ch.alpine.subare.book.ch08.maze;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class DynamazeTest {
  @Test
  void testSimple() {
    Dynamaze dynamaze = DynamazeHelper.original("maze2");
    assertEquals(dynamaze.startStates().length(), 1);
  }
}
