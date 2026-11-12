// code by jph
package ch.alpine.subare.net.run;

@FunctionalInterface
public interface NetTrainListener {
  boolean epoch(NetTrain netTrain, int epoch);
}
