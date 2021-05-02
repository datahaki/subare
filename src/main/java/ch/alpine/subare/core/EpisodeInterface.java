// code by jph
package ch.alpine.subare.core;

public interface EpisodeInterface {
  /** @return (s, a, r, s') */
  StepInterface step();

  /** @return true if current state is not terminal, else false */
  boolean hasNext();
}
