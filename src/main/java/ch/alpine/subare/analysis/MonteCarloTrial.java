// code by fluric
package ch.alpine.subare.analysis;

import ch.alpine.subare.core.api.TrueOnlineInterface;

public interface MonteCarloTrial extends TrueOnlineInterface {
  void executeBatch();
}
