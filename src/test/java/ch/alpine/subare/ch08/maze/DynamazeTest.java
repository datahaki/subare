// code by jph
package ch.alpine.subare.ch08.maze;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

public class DynamazeTest {
  @Test
  public void testSimple() {
    Dynamaze dynamaze = DynamazeHelper.original("maze2");
    assertEquals(dynamaze.startStates().length(), 1);
  }
}
