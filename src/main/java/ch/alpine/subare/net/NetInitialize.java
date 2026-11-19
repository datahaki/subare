// code by jph
package ch.alpine.subare.net;

import java.util.concurrent.ThreadLocalRandom;
import java.util.random.RandomGenerator;

import ch.alpine.subare.net.api.NetLayer;

public enum NetInitialize {
  ;
  public static void of(NetLayer netLayer, RandomGenerator randomGenerator) {
    NetExtract.flatten(netLayer) //
        .forEach(parametricLayer -> parametricLayer.initialize(randomGenerator));
  }

  public static void of(NetLayer netLayer) {
    of(netLayer, ThreadLocalRandom.current());
  }
}
