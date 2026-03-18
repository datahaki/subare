// code by jph
package ch.alpine.subare.mod;

import ch.alpine.tensor.Tensor;

public interface MonteCarloInterface extends TabularModel, TerminalInterface {
  /** @return states that are candidates to start an episode */
  Tensor startStates();
}
