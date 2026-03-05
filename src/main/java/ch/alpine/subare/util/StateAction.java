// code by jph
package ch.alpine.subare.util;

import ch.alpine.subare.api.StepRecord;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Unprotect;

public enum StateAction {
  ;
  /** @param state
   * @param action
   * @return */
  public static Tensor key(Tensor state, Tensor action) {
    return Unprotect.byRef(state, action).unmodifiable();
  }

  /** @param stepRecord
   * @return */
  public static Tensor key(StepRecord stepRecord) {
    return key(stepRecord.prevState(), stepRecord.action());
  }
}
