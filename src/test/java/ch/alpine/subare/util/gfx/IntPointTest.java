// code by jph
package ch.alpine.subare.util.gfx;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class IntPointTest {
  @Test
  void test() {
    IntPoint p1 = new IntPoint(2, 3);
    IntPoint p2 = new IntPoint(2, 3);
    assertEquals(p1, p2);
    assertEquals(p1.hashCode(), p2.hashCode());
  }
}
