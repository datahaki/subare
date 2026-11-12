// code by jph
package ch.alpine.subare.net;

import java.io.Serializable;
import java.util.stream.Stream;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.io.MathematicaFormat;

/** Residual layers help solve a problem called the vanishing gradient problem
 * and make it easier to train very deep neural networks.
 * 
 * Their main benefits are:
 * Better gradient flow: Gradients can travel through shortcut connections.
 * Easier training: Deep networks become easier to optimize.
 * Preserving information: The original input is not lost as it passes through layers.
 * Deeper networks: They allow neural networks to have many more layers. */
public class ResidualLayer implements NetLayer, Serializable {
  private final NetLayer netLayer;
  private Tensor outputCache;

  public ResidualLayer(NetLayer netLayer) {
    this.netLayer = netLayer;
  }

  @Override
  public Tensor forward(Tensor x) {
    return outputCache = netLayer.forward(x).add(x);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return netLayer.back(gradOutput).add(gradOutput);
  }

  @Override
  public void update(Scalar lr) {
    netLayer.update(lr);
  }

  @Override
  public Stream<Scalar> parameters() {
    return netLayer.parameters();
  }

  @Override
  public Stream<NetLayer> stream() {
    return netLayer.stream();
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("ResidualLayer", netLayer);
  }
}
