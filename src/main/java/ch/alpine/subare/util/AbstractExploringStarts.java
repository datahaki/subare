// code by jph
package ch.alpine.subare.util;

import ch.alpine.subare.api.EpisodeInterface;
import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.api.Policy;

/* package */ abstract class AbstractExploringStarts {
  private final MonteCarloInterface monteCarloInterface;
  private int batchIndex = -1; // incremented from constructor
  private Policy policy; // must be private
  private ExploringStartsBatch exploringStartBatch;
  private int episodeIndex = 0;

  protected AbstractExploringStarts(MonteCarloInterface monteCarloInterface) {
    this.monteCarloInterface = monteCarloInterface;
  }

  final void nextBatch() {
    ++batchIndex; // holds subsequent batch id that won't change during the next episodes
    exploringStartBatch = new ExploringStartsBatch(monteCarloInterface);
    policy = batchPolicy(batchIndex);
  }

  public final void nextEpisode() {
    EpisodeInterface episodeInterface = exploringStartBatch.nextEpisode(policy);
    // ---
    protected_nextEpisode(episodeInterface);
    // ---
    ++episodeIndex;
    if (!exploringStartBatch.hasNext())
      nextBatch();
  }

  public final int batchIndex() {
    return batchIndex;
  }

  public final int episodeIndex() {
    return episodeIndex;
  }

  public abstract void protected_nextEpisode(EpisodeInterface episodeInterface);

  /** @param batch = 0, 1, 2, ...
   * @return */
  public abstract Policy batchPolicy(int batch);
}
