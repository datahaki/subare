// code by jph
package ch.alpine.subare.api;

public interface EpisodeInterface {
  /** @return (s, a, r, s') */
  StepRecord step();

  /** @return true if current state is not terminal, else false */
  boolean hasNext();
}
