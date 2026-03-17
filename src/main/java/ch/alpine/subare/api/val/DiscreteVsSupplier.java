// code by jph
package ch.alpine.subare.api.val;

import ch.alpine.subare.util.DiscreteVs;

@FunctionalInterface
public interface DiscreteVsSupplier {
  DiscreteVs vs();
}
