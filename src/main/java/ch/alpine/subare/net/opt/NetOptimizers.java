// code by jph
package ch.alpine.subare.net.opt;

import ch.alpine.tensor.Scalar;

public enum NetOptimizers {
  SIMPLE {
    @Override
    public NetOptimizerFactory with(Scalar eta, Scalar... scalar) {
      return netLayer -> new GradientDescentOptimizer(netLayer, eta);
    }
  },
  MOMENT {
    @Override
    public NetOptimizerFactory with(Scalar eta, Scalar... scalar) {
      return netLayer -> new MomentumOptimizer(netLayer, true, eta, scalar[0]);
    }
  },
  MOMENT_W {
    @Override
    public NetOptimizerFactory with(Scalar eta, Scalar... scalar) {
      return netLayer -> new MomentumOptimizer(netLayer, false, eta, scalar[0]);
    }
  },
  ADAM {
    @Override
    public NetOptimizerFactory with(Scalar eta, Scalar... scalar) {
      return netLayer -> new AdamOptimizer(netLayer, true, eta, scalar[0], scalar[1]);
    }
  },
  ADAM_W {
    @Override
    public NetOptimizerFactory with(Scalar eta, Scalar... scalar) {
      return netLayer -> new AdamOptimizer(netLayer, false, eta, scalar[0], scalar[1]);
    }
  };

  /** @param eta
   * @param scalar
   * @return */
  public abstract NetOptimizerFactory with(Scalar eta, Scalar... scalar);
}
