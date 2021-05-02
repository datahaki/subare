// code by jph
package ch.alpine.subare.core.util;

import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.QsaInterfaceSupplier;
import ch.alpine.subare.core.StateActionCounterSupplier;
import ch.alpine.tensor.Tensor;

public interface PolicyExt extends Policy, QsaInterfaceSupplier, StateActionCounterSupplier {
  /** @param state
   * @return vector of actions that are equally optimal */
  Tensor getBestActions(Tensor state);

  PolicyExt copy();
}
