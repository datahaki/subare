package ch.alpine.subare.net.opt;

import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Map.Entry;

import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public class NetMomentumGradientDescent implements NetOptimizer {
  static interface Updater {
  }

  static class LLUpdater implements Updater {
    final LinearLayer linearLayer;
    Tensor vW;
    Tensor vb;

    public LLUpdater(LinearLayer linearLayer) {
      this.linearLayer = linearLayer;
      vW = linearLayer.W.maps(Scalar::zero);
      vb = linearLayer.b.maps(Scalar::zero);
    }
  }

  private final Map<NetLayer, Updater> map = new IdentityHashMap<>();
  private final Scalar beta;
  private final Scalar eta;

  public NetMomentumGradientDescent(NetLayer netLayer, Scalar beta, Scalar eta) {
    for (NetLayer _layer : netLayer.stream().toList()) {
      if (_layer instanceof LinearLayer linearLayer)
        map.put(_layer, new LLUpdater(linearLayer));
    }
    this.beta = beta;
    this.eta = eta;
  }

  @Override
  public void update() {
    for (Entry<NetLayer, Updater> entry : map.entrySet()) {
      NetLayer netLayer = entry.getKey();
      if (netLayer instanceof LinearLayer linearLayer) {
        LLUpdater llUpdater = (LLUpdater) entry.getValue();
        llUpdater.vW = llUpdater.vW.multiply(beta).add(linearLayer.gW);
        linearLayer.W = linearLayer.W.subtract(llUpdater.vW.multiply(eta));
        // ---
        llUpdater.vb = llUpdater.vb.multiply(beta).add(linearLayer.gb);
        linearLayer.b = linearLayer.b.subtract(llUpdater.vb.multiply(eta));
      }
    }
  }
}
