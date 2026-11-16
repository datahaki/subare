// code by jph
package ch.alpine.subare.net.run;

public record MaxEpochListener(int max_epochs) implements NetTrainListener {
  @Override
  public boolean epoch(NetTrain netTrain) {
    return netTrain.getTotalEpochs() < max_epochs;
  }
}
