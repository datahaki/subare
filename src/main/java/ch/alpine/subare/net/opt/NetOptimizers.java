package ch.alpine.subare.net.opt;

import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.tensor.Scalar;

public enum NetOptimizers {
  SIMPLE {
    @Override
    public NetOptimizer create(NetLayer netLayer, Scalar eta, Scalar... scalar) {
      return new NetSimpleGradientDescent(netLayer, eta);
    }
  },
  MOMENT {
    @Override
    public NetOptimizer create(NetLayer netLayer, Scalar eta, Scalar... scalar) {
      return new NetMomentumGradientDescent(netLayer, eta, scalar[0]);
    }
  },
  ADAM {
    @Override
    public NetOptimizer create(NetLayer netLayer, Scalar eta, Scalar... scalar) {
      return new AdamOptimizer(netLayer, eta, scalar[0], scalar[1]);
    }
  };

  public abstract NetOptimizer create(NetLayer netLayer, Scalar eta, Scalar... scalar);
}
