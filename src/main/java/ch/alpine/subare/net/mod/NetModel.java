// code by jph
package ch.alpine.subare.net.mod;

import java.io.Serializable;

import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.los.LossFunction;

public record NetModel(NetLayer netLayer, LossFunction lossFunction) implements Serializable {
}
