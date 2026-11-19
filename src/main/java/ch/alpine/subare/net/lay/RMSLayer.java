// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.Objects;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Numel;
import ch.alpine.tensor.nrm.Vector2NormSquared;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.sca.pow.Sqrt;

// TODO UNFINISHED UNTESTED WORK IN PROGRESS !!!
public class RMSLayer implements NetLayer, Serializable {
  private static final Scalar EPS = RealScalar.of(1E-6);
  public Tensor gamma = null;
  private transient Scalar rmsx;
  private transient Tensor xhat;

  @Override
  public Tensor forward(Tensor x) {
    if (Objects.isNull(gamma))
      gamma = x.maps(Scalar::one);
    int d = Numel.of(x);
    rmsx = Sqrt.FUNCTION.apply(Vector2NormSquared.of(x).divide(RealScalar.of(d)).add(EPS));
    IO.println(rmsx);
    xhat = x.divide(rmsx);
    return Entrywise.mul().apply(xhat, gamma);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    // TODO Auto-generated method stub
    return null;
  }
}
