// code by jph
package ch.alpine.subare.core;

@FunctionalInterface
public interface EpisodeDigest {
  /** @param episodeInterface */
  void digest(EpisodeInterface episodeInterface);
}