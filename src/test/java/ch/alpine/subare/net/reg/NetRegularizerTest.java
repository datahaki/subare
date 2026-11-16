// code by jph
package ch.alpine.subare.net.reg;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class NetRegularizerTest {
  @Test
  void test() {
    String string = NetRegularizer.NONE.toString();
    assertEquals(string, "NoReg");
  }
}
