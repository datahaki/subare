// code by jph and fluric
package ch.alpine.subare.api;

import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.tensor.Tensor;

/** interface to indicate how often the state-action pair is visited during learning
 * 
 * the state-action count is updated for every digested step */
public interface StateActionCounter extends StepDigest {
  /** @param key same as listed in {@link DiscreteQsa#keys()}
   * @return number of updates of qsa value for given state-action pair due to learning */
  int stateActionCount(Tensor key);

  /** @param state same as listed in {@link DiscreteQsa#keys()}
   * @return number of updates of qsa value for given state due to learning */
  int stateCount(Tensor state);

  /** function exists to remove the initialization bias
   * 
   * @param key that contains the state-action pair
   * @return whether given (state, action) pair has already been encountered by learning rate */
  boolean isEncountered(Tensor key);
}
