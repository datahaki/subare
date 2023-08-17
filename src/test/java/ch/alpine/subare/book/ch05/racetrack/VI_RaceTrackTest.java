// code by jph
package ch.alpine.subare.book.ch05.racetrack;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.ext.HomeDirectory;

class VI_RaceTrackTest {
  @Test
  void testSimple() throws Exception {
    File file = HomeDirectory.Pictures(getClass().getSimpleName() + ".gif");
    assertFalse(file.exists());
    VI_RaceTrack.make("track2", 4, file);
    assertTrue(file.exists());
    file.delete();
  }
}
