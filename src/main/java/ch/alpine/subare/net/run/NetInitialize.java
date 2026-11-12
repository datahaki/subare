package ch.alpine.subare.net.run;

import java.util.concurrent.ThreadLocalRandom;
import java.util.random.RandomGenerator;

import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.subare.net.lay.NetLayer;

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
