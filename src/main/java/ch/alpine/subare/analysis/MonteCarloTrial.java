// code by fluric
package ch.alpine.subare.analysis;

import ch.alpine.subare.api.pol.TrueOnlineInterface;

public interface MonteCarloTrial extends TrueOnlineInterface {
  void executeBatch();
}
