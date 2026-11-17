// code by jph
package ch.alpine.subare.net.opt;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;

import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.subare.net.lay.LinearLayer.AffineTuple;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.tensor.Scalar;

/** simple gradient descent */
public class NetSimpleGradientDescent implements NetOptimizer, Serializable {
  private final Scalar eta;
  private final List<LinearLayer> linearLayers = new LinkedList<>();

  public NetSimpleGradientDescent(NetLayer netLayer, Scalar eta) {
    this.eta = eta;
    for (NetLayer _layer : netLayer.stream().toList()) {
      if (_layer instanceof LinearLayer linearLayer)
        linearLayers.add(linearLayer);
    }
  }

  @Override
  public void update() {
    for (LinearLayer linearLayer : linearLayers) {
      AffineTuple affineTuple = linearLayer.Wb();
      linearLayer.setParam( //
          affineTuple.W().add(linearLayer.gW.multiply(eta.negate())), //
          affineTuple.b().add(linearLayer.gb.multiply(eta.negate())));
    }
  }
}
