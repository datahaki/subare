// code by jph
package ch.alpine.subare.core;

import ch.alpine.subare.core.util.DiscreteVs;

@FunctionalInterface
public interface DiscreteVsSupplier {
  DiscreteVs vs();
}