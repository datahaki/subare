// code by jph
package ch.alpine.subare.net.lay;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.sca.exp.LogisticSigmoid;

class SiluLayer extends ElementwiseLayer {
  transient Tensor sig_cache;

  @Override
  public Tensor forward(Tensor x) {
    sig_cache = (logits = x).maps(LogisticSigmoid.FUNCTION);
    return Entrywise.mul().apply(x, sig_cache);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    Tensor diff = sig_cache.maps(s -> RealScalar.ONE.subtract(s));
    Tensor step = Entrywise.mul().apply(logits, diff).maps(RealScalar.ONE::add);
    return Entrywise.mul().apply(gradOutput, Entrywise.mul().apply(sig_cache, step));
  }

  @Override
  protected String title() {
    return "SiLU";
  }
}
