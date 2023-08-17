// code by jph
package ch.alpine.subare.api;

@FunctionalInterface
public interface EpisodeDigest {
  /** @param episodeInterface */
  void digest(EpisodeInterface episodeInterface);
}
