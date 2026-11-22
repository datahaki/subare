// code by jph
package ch.alpine.subare.net.opt;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.tensor.Scalar;

public enum NetOptimizers {
  SIMPLE {
    @Override
    public NetOptimizer create(NetLayer netLayer, Scalar eta, Scalar... scalar) {
      return new GradientDescentOptimizer(netLayer, eta);
    }
  },
  MOMENT {
    @Override
    public NetOptimizer create(NetLayer netLayer, Scalar eta, Scalar... scalar) {
      return new MomentumOptimizer(netLayer, true, eta, scalar[0]);
    }
  },
  MOMENT_W {
    @Override
    public NetOptimizer create(NetLayer netLayer, Scalar eta, Scalar... scalar) {
      return new MomentumOptimizer(netLayer, false, eta, scalar[0]);
    }
  },
  ADAM {
    @Override
    public NetOptimizer create(NetLayer netLayer, Scalar eta, Scalar... scalar) {
      return new AdamOptimizer(netLayer, true, eta, scalar[0], scalar[1]);
    }
  },
  ADAM_W {
    @Override
    public NetOptimizer create(NetLayer netLayer, Scalar eta, Scalar... scalar) {
      return new AdamOptimizer(netLayer, false, eta, scalar[0], scalar[1]);
    }
  };

  public abstract NetOptimizer create(NetLayer netLayer, Scalar eta, Scalar... scalar);
}
