// code by jph
package ch.alpine.subare.net.opt;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetLayers;
import ch.alpine.subare.net.NetOptimizable;
import ch.alpine.tensor.Scalar;

abstract class AbstractNetOptimizer implements NetOptimizer {
  private final List<Runnable> list;
  final boolean weightDecay;
  private final Scalar eta;
  int count = 0;

  public AbstractNetOptimizer(NetLayer netLayer, boolean weightDecay, Scalar eta) {
    list = NetLayers.trainableLayers(netLayer) //
        .map(this::create) //
        .collect(Collectors.toCollection(ArrayList::new));
    this.weightDecay = weightDecay;
    this.eta = eta;
  }

  abstract Runnable create(NetOptimizable netOptimizable);

  @Override
  public final void update() {
    list.forEach(Runnable::run);
    ++count;
  }

  @Override
  public final boolean weightDecay() {
    return weightDecay;
  }

  Scalar eta() {
    return eta;
  }
}
