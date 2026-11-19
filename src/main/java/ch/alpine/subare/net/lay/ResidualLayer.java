// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.random.RandomGenerator;

import ch.alpine.subare.net.api.NetLayer;
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
  public void initialize(RandomGenerator randomGenerator) {
    netLayer.initialize(randomGenerator);
  }

  /** @return */
  @Override
  public Tensor parameters() {
    return netLayer.parameters();
  }

  /** @param decayInLoss whether to include the gradient from the L2 (!) regularization,
   * i.e. "weight decay", in the gradient descent
   * @return */
  @Override
  public Tensor gradient(boolean decayInLoss) {
    return netLayer.gradient(decayInLoss);
  }

  /** @param decayInLoss whether the gradient from the L2 (!) regularization was considered
   * in the gradient descent
   * @param delta
   * @param eta */
  @Override
  public void increment(boolean decayInLoss, Tensor delta, Scalar eta) {
    netLayer.increment(decayInLoss, delta, eta);
  }

  /** @return contribution to loss by weight regularization */
  @Override
  public Scalar reg_loss(boolean decayInLoss) {
    return netLayer.reg_loss(decayInLoss);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("ResidualLayer", netLayer);
  }
}
