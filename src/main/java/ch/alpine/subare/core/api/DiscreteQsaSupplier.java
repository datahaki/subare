// code by jph
package ch.alpine.subare.core.api;

import ch.alpine.subare.core.util.DiscreteQsa;

@FunctionalInterface
public interface DiscreteQsaSupplier {
  DiscreteQsa qsa();
}
