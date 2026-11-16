// code by jph
package ch.alpine.subare.net.opt;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;

import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

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
    for (LinearLayer ll : linearLayers) {
      Tensor nW = ll.W().add(ll.gW.multiply(eta.negate()));
      Tensor nb = ll.b().add(ll.gb.multiply(eta.negate()));
      ll.setParam(nW, nb);
    }
  }
}
