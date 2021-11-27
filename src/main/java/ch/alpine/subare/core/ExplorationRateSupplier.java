// code by fluric
package ch.alpine.subare.core;

import ch.alpine.subare.core.util.ExplorationRate;

@FunctionalInterface
public interface ExplorationRateSupplier {
  ExplorationRate explorationRate();
}