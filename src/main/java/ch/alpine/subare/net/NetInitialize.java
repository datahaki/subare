// code by jph
package ch.alpine.subare.net;

import java.util.concurrent.ThreadLocalRandom;

import ch.alpine.subare.net.api.NetLayer;

public enum NetInitialize {
  ;
  public static void of(NetLayer netLayer) {
    netLayer.initialize(ThreadLocalRandom.current());
  }
}
