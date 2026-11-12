package ch.alpine.subare.net;

import java.util.concurrent.ThreadLocalRandom;
import java.util.random.RandomGenerator;

public enum NetInitialize {
  ;
  public static void of(NetLayer netLayer, RandomGenerator randomGenerator) {
    netLayer.stream().forEach(l -> check(l, randomGenerator));
  }

  public static void of(NetLayer netLayer) {
    of(netLayer, ThreadLocalRandom.current());
  }

  private static void check(NetLayer netLayer, RandomGenerator randomGenerator) {
    if (netLayer instanceof LinearLayer linearLayer) {
      linearLayer.initialize(randomGenerator);
    }
  }
}
