// code by jph
package ch.alpine.subare.math;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class Float16Test {
  @Test
  void test() {
    short val = Float16.floatToHalf((float) Math.PI);
    float halfToFloat = Float16.halfToFloat(val);
    String string = "" + halfToFloat;
    assertTrue(string.startsWith("3.14"));
  }
}
