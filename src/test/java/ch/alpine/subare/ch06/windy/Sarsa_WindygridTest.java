// code by jph
package ch.alpine.subare.ch06.windy;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.td.SarsaType;

class Sarsa_WindygridTest {
  @Test
  public void testSimple() throws Exception {
    for (SarsaType sarsaType : SarsaType.values()) {
      Sarsa_Windygrid.handle(sarsaType, 10);
      File file = Sarsa_Windygrid.getFileQsa(sarsaType);
      assertTrue(file.isFile());
      file.delete();
    }
  }
}
