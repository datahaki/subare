// code by jph
package ch.alpine.subare.net;

@FunctionalInterface
public interface NetTrainListener {
  boolean epoch(NetTrain netTrain, int epoch);
}
