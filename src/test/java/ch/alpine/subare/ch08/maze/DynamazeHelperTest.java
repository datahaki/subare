// code by jph
package ch.alpine.subare.ch08.maze;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Dimensions;

class DynamazeHelperTest {
  @Test
  public void testMaze2() {
    Tensor image = DynamazeHelper.load("maze2");
    assertEquals(Dimensions.of(image), List.of(6, 9, 4));
  }

  @Test
  public void testMaze5() {
    Tensor image = DynamazeHelper.load("maze5");
    assertEquals(Dimensions.of(image), List.of(32, 16, 4));
  }
}
