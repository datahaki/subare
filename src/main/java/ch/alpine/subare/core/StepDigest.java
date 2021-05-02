// code by jph
package ch.alpine.subare.core;

/** interface is implemented by temporal difference algorithms */
@FunctionalInterface
public interface StepDigest {
  /** update based on a single step of an episode
   * 
   * @param stepInterface */
  void digest(StepInterface stepInterface);
}
