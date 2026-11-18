// code by jph
package ch.alpine.subare.net.reg;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class NetRegularizerTest {
  @Test
  void test() {
    String string = NoWeightDecay.INSTANCE.toString();
    assertEquals(string, "NoReg");
  }
}
