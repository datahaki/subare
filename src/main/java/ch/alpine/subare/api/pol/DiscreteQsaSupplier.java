// code by jph
package ch.alpine.subare.api.pol;

import ch.alpine.subare.util.DiscreteQsa;

@FunctionalInterface
public interface DiscreteQsaSupplier {
  DiscreteQsa qsa();
}
