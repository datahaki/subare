// code by jph
package ch.alpine.subare.net.opt;

import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.tensor.Scalar;

/** simple gradient descent */
class GradientDescentOptimizer extends AbstractNetOptimizer {
  public GradientDescentOptimizer(NetLayer netLayer, Scalar eta) {
    super(netLayer, true, eta);
  }

  @Override
  Runnable create(NetOptimizable netOptimizable) {
    // TODO eta() should not never be zero here!
    return () -> netOptimizable.step(weightDecay, netOptimizable.gradient(weightDecay), eta());
  }
}
