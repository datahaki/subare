// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.ethz.idsc.subare.core.StepInterface;

public enum StateAction {
  ;
  /** @param state
   * @param action
   * @return */
  public static Tensor key(Tensor state, Tensor action) {
    return Tensors.of(state, action);
  }

  /** @param stepInterface
   * @return */
  public static Tensor key(StepInterface stepInterface) {
    return key(stepInterface.prevState(), stepInterface.action());
  }

  public static Tensor getState(Tensor key) {
    return key.get(0);
  }

  public static Tensor getAction(Tensor key) {
    return key.get(1);
  }
}
