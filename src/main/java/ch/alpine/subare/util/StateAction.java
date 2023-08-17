// code by jph
package ch.alpine.subare.util;

import ch.alpine.subare.api.StepRecord;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

public enum StateAction {
  ;
  /** @param state
   * @param action
   * @return */
  public static Tensor key(Tensor state, Tensor action) {
    return Tensors.of(state, action);
  }

  /** @param stepRecord
   * @return */
  public static Tensor key(StepRecord stepRecord) {
    return key(stepRecord.prevState(), stepRecord.action());
  }

  public static Tensor getState(Tensor key) {
    return key.get(0);
  }

  public static Tensor getAction(Tensor key) {
    return key.get(1);
  }
}
