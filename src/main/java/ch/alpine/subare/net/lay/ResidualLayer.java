// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.stream.Stream;

import ch.alpine.subare.net.api.NetLayer;
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
// TODO UNFINISHED UNTESTED WORK IN PROGRESS !!!
public class ResidualLayer implements NetLayer, Serializable {
  private final NetLayer netLayer;

  public ResidualLayer(NetLayer netLayer) {
    this.netLayer = netLayer;
  }

  @Override
  public Tensor forward(Tensor x) {
    return netLayer.forward(x).add(x);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    return netLayer.forwardBatch(X).add(X);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return netLayer.back(gradOutput).add(gradOutput);
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    return netLayer.backBatch(gradOutput).add(gradOutput);
  }

  @Override
  public Stream<NetLayer> stream() {
    // FIXME this does not make sense
    return netLayer.stream();
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("ResidualLayer", netLayer);
  }
}
