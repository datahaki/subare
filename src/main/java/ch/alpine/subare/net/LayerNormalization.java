package ch.alpine.subare.net;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.red.Variance;
import ch.alpine.tensor.sca.pow.Sqrt;

public class LayerNormalization implements Layer {
  private static final Scalar EPS = RealScalar.of(1e-5);
  // trainable params
  private Tensor gamma;
  private Tensor beta;
  // gradients
  private Tensor dGamma;
  private Tensor dBeta;
  // cache
  private Tensor input;
  private Tensor normalized;

  public LayerNormalization(int dim) {
    gamma = Array.same(RealScalar.ONE, dim);
    beta = Array.zeros(dim);
  }

  @Override
  public Tensor forward(Tensor x) {
    input = x;
    Scalar mean = Mean.ofVector(x);
    Scalar variance = Variance.ofVector(x);
    // TODO / dim instead of / d-1
    Scalar std = Sqrt.FUNCTION.apply(variance.add(EPS));
    x.maps(mean.negate()::add).divide(std);
    return Entrywise.mul().apply(gamma, normalized).add(beta);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    dGamma = dGamma.add( //
        Entrywise.mul().apply(gradOutput, normalized));
    dBeta = dBeta.add(gradOutput);
    Scalar sumGrad = (Scalar) gradOutput.dot(gamma);
    // FIXME
    // return gradInput;
    return null;
  }

  @Override
  public void update() {
    // FIXME
  }

  @Override
  public Tensor error(Tensor y) {
    // TODO Auto-generated method stub
    return null;
  }
}
