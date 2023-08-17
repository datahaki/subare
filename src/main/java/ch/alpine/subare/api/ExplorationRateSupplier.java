// code by fluric
package ch.alpine.subare.api;

import ch.alpine.subare.util.ExplorationRate;

@FunctionalInterface
public interface ExplorationRateSupplier {
  ExplorationRate explorationRate();
}
