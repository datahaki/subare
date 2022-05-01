// code by jph
package ch.alpine.subare.ch06.windy;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

class WindygridTest {
  @Test
  public void testWindy() {
    Windygrid windyGrid = Windygrid.createFour();
    Tensor state = Tensors.vector(6, 0);
    windyGrid.actions(state);
  }

  @Test
  public void testRepmat() {
    Windygrid windyGrid = Windygrid.createFour();
    Tensor left = Tensors.vector(-1, 0);
    Tensor up = Tensors.vector(0, 1);
    Tensor state = Windygrid.GOAL.add(up);
    Tensor dest = windyGrid.move(state, left);
    assertEquals(dest, Tensors.vector(6, 2));
    windyGrid.actions(state);
  }
}
