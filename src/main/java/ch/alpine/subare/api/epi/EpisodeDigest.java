// code by jph
package ch.alpine.subare.api.epi;

import ch.alpine.subare.api.EpisodeInterface;

@FunctionalInterface
public interface EpisodeDigest {
  /** @param episodeInterface */
  void digest(EpisodeInterface episodeInterface);
}
