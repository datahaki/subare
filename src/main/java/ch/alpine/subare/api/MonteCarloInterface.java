// code by jph
package ch.alpine.subare.api;

import ch.alpine.tensor.Tensor;

public interface MonteCarloInterface extends DiscreteModel, SampleModel, TerminalInterface {
  /** @return states that are candidates to start an episode */
  Tensor startStates();
}
