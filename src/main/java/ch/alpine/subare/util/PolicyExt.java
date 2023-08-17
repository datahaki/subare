// code by jph
package ch.alpine.subare.util;

import ch.alpine.subare.api.Policy;
import ch.alpine.subare.api.QsaInterfaceSupplier;
import ch.alpine.subare.api.StateActionCounterSupplier;
import ch.alpine.tensor.Tensor;

public interface PolicyExt extends Policy, QsaInterfaceSupplier, StateActionCounterSupplier {
  /** @param state
   * @return vector of actions that are equally optimal */
  Tensor getBestActions(Tensor state);

  PolicyExt copy();
}
