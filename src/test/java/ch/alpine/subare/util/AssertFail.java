// code by jph
package ch.alpine.subare.util;

import static org.junit.jupiter.api.Assertions.fail;

import java.util.Objects;

public enum AssertFail {
  ;
  public static void of(Runnable runnable) {
    Objects.requireNonNull(runnable);
    try {
      runnable.run();
      fail();
    } catch (Exception exception) {
      // ---
    }
  }
}
