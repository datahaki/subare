// code by jph
package ch.alpine.subare.epi;

@FunctionalInterface
public interface EpisodeDigest {
  /** @param episodeInterface */
  void digest(EpisodeInterface episodeInterface);
}
