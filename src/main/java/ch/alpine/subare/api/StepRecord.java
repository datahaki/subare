// code by jph
package ch.alpine.subare.api;

import java.io.Serializable;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** record provides the four entries (s, a, r, s')
 * 
 * previous state
 * action that was taken to reach next state */
public record StepRecord(Tensor prevState, Tensor action, Scalar reward, Tensor nextState) implements Serializable {
}
