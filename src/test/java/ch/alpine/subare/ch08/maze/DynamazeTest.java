// code by jph
package ch.alpine.subare.ch08.maze;

import junit.framework.TestCase;

public class DynamazeTest extends TestCase {
  public void testSimple() {
    Dynamaze dynamaze = DynamazeHelper.original("maze2");
    assertEquals(dynamaze.startStates().length(), 1);
  }
}
