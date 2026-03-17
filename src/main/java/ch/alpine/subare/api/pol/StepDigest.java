// code by jph
package ch.alpine.subare.api.pol;

import ch.alpine.subare.api.StepRecord;

/** interface is implemented by temporal difference algorithms */
@FunctionalInterface
public interface StepDigest {
  /** update based on a single step of an episode
   * 
   * @param stepRecord */
  void digest(StepRecord stepRecord);
}
