// code by jph
package ch.alpine.subare.book.ch05.racetrack;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class VI_RaceTrackTest {
  @TempDir
  Path tempDir;

  @Test
  void testSimple() throws Exception {
    Path file = tempDir.resolve(getClass().getSimpleName() + ".gif");
    VI_RaceTrack.make("track2", 4, file);
    assertTrue(Files.exists(file));
  }
}
