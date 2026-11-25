package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.NetLayer;
import ch.alpine.subare.net.err.AbstractProbBinaryCrossEntropy;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.sca.exp.LogisticSigmoid;

/** intended for use as last layer in a network
 * otherwise use {@link ElementwiseLayers#SIGMOID}
 * 
 * {@link SigmoidLayer} provides a loss function */
public class SigmoidLayer implements NetLayer, Serializable {
  private transient Tensor logits;
  private transient Tensor activations;

  /** @param x input tensor (scalar or tensor of pre-activations)
   * @return probability tensor in (0, 1) */
  @Override
  public Tensor forward(Tensor x) {
    logits = x;
    return activations = x.maps(LogisticSigmoid.FUNCTION);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    logits = X;
    return activations = X.maps(LogisticSigmoid.FUNCTION);
  }

  public LossFunction lossFunction() {
    return new AbstractProbBinaryCrossEntropy() {
      @Override
      protected Tensor logits() {
        return logits;
      }
    };
  }

  /** @param gradOutput upstream gradient dL/dp
   * @return gradient dL/dz = (dL/dp) * p * (1 - p) */
  @Override
  public Tensor back(Tensor gradOutput) {
    return Entrywise.mul().apply(gradOutput, sigmoidDerivative(activations));
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    return Entrywise.mul().apply(gradOutput, sigmoidDerivative(activations));
  }

  private static Tensor sigmoidDerivative(Tensor p) {
    return p.maps(prob -> prob.multiply(RealScalar.ONE.subtract(prob)));
  }

  @Override
  public String toString() {
    return "SigmoidLayer";
  }
}
