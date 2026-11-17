// code by jph
package ch.alpine.subare.net.opt;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import ch.alpine.subare.net.lay.LinearLayer;
import ch.alpine.subare.net.lay.LinearLayer.AffineTuple;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

public class NetMomentumGradientDescent implements NetOptimizer, Serializable {
  private static interface Updater {
    void update(Scalar beta, Scalar eta);
  }

  private static class LLUpdater implements Updater {
    private final LinearLayer linearLayer;
    // velocity
    Tensor vW;
    Tensor vb;

    public LLUpdater(LinearLayer linearLayer) {
      this.linearLayer = linearLayer;
      AffineTuple affineTuple = linearLayer.Wb();
      vW = affineTuple.W().maps(Scalar::zero);
      vb = affineTuple.b().maps(Scalar::zero);
    }

    @Override
    public void update(Scalar beta, Scalar eta) {
      AffineTuple affineTuple = linearLayer.Wb();
      //
      Tensor nW;
      {
        Tensor gW = linearLayer.gradientW();
        vW = vW.multiply(beta).add(gW);
        nW = affineTuple.W().subtract(vW.multiply(eta));
      } // ---
      Tensor nb;
      {
        Tensor gb = linearLayer.gb;
        vb = vb.multiply(beta).add(gb);
        nb = affineTuple.b().subtract(vb.multiply(eta));
      }
      linearLayer.setParam(nW, nb);
    }
  }

  private final List<Updater> list;
  private final Scalar beta;
  private final Scalar eta;

  public NetMomentumGradientDescent(NetLayer netLayer, Scalar beta, Scalar eta) {
    list = netLayer.stream().flatMap(l -> buildMap(l).stream()).collect(Collectors.toCollection(ArrayList::new));
    this.beta = beta;
    this.eta = eta;
  }

  private static Optional<Updater> buildMap(NetLayer netLayer) {
    if (netLayer instanceof LinearLayer linearLayer)
      return Optional.of(new LLUpdater(linearLayer));
    return Optional.empty();
  }

  @Override
  public void update() {
    list.forEach(u -> u.update(beta, eta));
  }
}
