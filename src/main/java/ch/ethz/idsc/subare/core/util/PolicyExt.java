// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.alpine.tensor.Tensor;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterfaceSupplier;
import ch.ethz.idsc.subare.core.StateActionCounterSupplier;

public interface PolicyExt extends Policy, QsaInterfaceSupplier, StateActionCounterSupplier {
  /** @param state
   * @return vector of actions that are equally optimal */
  Tensor getBestActions(Tensor state);

  PolicyExt copy();
}
