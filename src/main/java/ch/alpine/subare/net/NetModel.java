// code by jph
package ch.alpine.subare.net;

import java.io.Serializable;

import ch.alpine.subare.net.err.SquaredErrorLoss;

public record NetModel(NetLayer netLayer, LossFunction lossFunction) implements Serializable {
  public static NetModel squaredErrorLoss(NetLayer netLayer) {
    return new NetModel(netLayer, SquaredErrorLoss.INSTANCE);
  }
}
