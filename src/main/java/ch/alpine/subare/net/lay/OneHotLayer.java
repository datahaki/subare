// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.api.NetLayer;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.mat.IdentityMatrix;

/** Embeds integers between 0 and (v-1) into a continuous vector space of dimension d.
 * 
 * Backpropagation stops at the embedding layer!
 * 
 * inspired by
 * <a href="https://reference.wolfram.com/language/ref/EmbeddingLayer.html">EmbeddingLayer</a> */
public class OneHotLayer implements NetLayer, Serializable {
  private final Tensor[] id;

  public OneHotLayer(int v) {
    id = IdentityMatrix.of(v).stream() //
        .map(Tensor::unmodifiable).toArray(Tensor[]::new);
  }

  @Override
  public Tensor forward(Tensor x) {
    VectorQ.requireLength(x, 1);
    // returns dimensions: [x.length() * d]
    return id[Scalars.intValueExact(x.Get(0))];
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    return Tensor.of(X.stream().map(x -> id[Scalars.intValueExact(x.Get(0))]));
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return null;
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    return null;
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("OneHotLayer", id.length);
  }
}
