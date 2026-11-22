// code by jph
package ch.alpine.subare.net.opt;

import java.util.Objects;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.NetOptimizable;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

class MomentumOptimizer extends AbstractNetOptimizer {
  private class Updater implements Runnable {
    private final NetOptimizable netOptimizable;
    Tensor v = null;

    public Updater(NetOptimizable netOptimizable) {
      this.netOptimizable = netOptimizable;
    }

    @Override
    public void run() {
      Tensor g = netOptimizable.gradient(weightDecay);
      v = Objects.isNull(v) ? g : v.multiply(beta).add(g);
      netOptimizable.step(weightDecay, v, eta());
    }
  }

  private final Scalar beta;

  public MomentumOptimizer(NetLayer netLayer, boolean weightDecay, Scalar eta, Scalar beta) {
    super(netLayer, weightDecay, eta);
    this.beta = beta;
  }

  @Override
  Runnable create(NetOptimizable netOptimizable) {
    return new Updater(netOptimizable);
  }
}
