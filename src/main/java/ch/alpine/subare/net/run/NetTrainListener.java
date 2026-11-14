// code by jph
package ch.alpine.subare.net.run;

@FunctionalInterface
public interface NetTrainListener {
  /** @param netTrain
   * @param epoch
   * @return whether to continue training */
  boolean epoch(NetTrain netTrain, int epoch);
}
