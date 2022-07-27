// code by jph
package ch.alpine.subare.core.util;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class TransitionTrackerTest {
  @Test
  void testSimple() {
    Map<Integer, Integer> map = new HashMap<>();
    map.merge(10, 101, Integer::sum);
    map.merge(10, 1, Integer::sum);
    map.merge(10, 1, Integer::sum);
    assertEquals(map.get(10).intValue(), 103);
  }
}
