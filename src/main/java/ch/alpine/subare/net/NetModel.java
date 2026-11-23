// code by jph
package ch.alpine.subare.net;

import java.io.Serializable;

public record NetModel(NetLayer netLayer, LossFunction lossFunction) implements Serializable {
}
