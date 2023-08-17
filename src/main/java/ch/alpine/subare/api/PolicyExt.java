// code by jph
package ch.alpine.subare.api;

import ch.alpine.tensor.Tensor;

public interface PolicyExt extends Policy, QsaInterfaceSupplier, StateActionCounterSupplier {
  /** @param state
   * @return vector of actions that are equally optimal */
  Tensor getBestActions(Tensor state);

  PolicyExt copy();
}
