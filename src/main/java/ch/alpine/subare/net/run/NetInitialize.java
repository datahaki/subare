// code by jph
package ch.alpine.subare.net.run;

import java.util.concurrent.ThreadLocalRandom;
import java.util.random.RandomGenerator;

import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.ParametricLayer;

public enum NetInitialize {
  ;
  public static void of(NetLayer netLayer, RandomGenerator randomGenerator) {
    netLayer.stream() //
        .filter(ParametricLayer.class::isInstance) //
        .map(ParametricLayer.class::cast) //
        .forEach(parametricLayer -> parametricLayer.initialize(randomGenerator));
  }

  public static void of(NetLayer netLayer) {
    of(netLayer, ThreadLocalRandom.current());
  }
}
