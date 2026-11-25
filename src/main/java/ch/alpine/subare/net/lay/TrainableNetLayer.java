// code by jph
package ch.alpine.subare.net.lay;

import java.util.random.RandomGenerator;

import ch.alpine.subare.net.opt.NetOptimizable;
import ch.alpine.subare.net.opt.NetOptimizer;
import ch.alpine.subare.net.opt.NetOptimizers;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** intermediate interface helps to distinguish
 * layers with trainable parameters from param-free layers
 * 
 * param-free layers could still extend from here
 * initialize = do nothing
 * parameters = {}
 * gradient = {}
 * increment = do nothing */
public interface TrainableNetLayer extends NetLayer, NetOptimizable {
  /** Parametric Layers:
   * Layers that contain internal variables (weights and biases) that are adjusted and
   * optimized during training (e.g., Dense / Linear layers, convolutional layers).
   * 
   * Layers whose parameters receive gradients during back propagation and are updated
   * by the optimizer. */
  void initialize(RandomGenerator randomGenerator);

  /** e.g. for dropout layer
   * 
   * @param isTraining */
  default void setTraining(boolean isTraining) {
    // TODO term "inference" !?
  }

  /** @return */
  Tensor parameters();

  /** note:
   * for certain {@link NetOptimizer} the weight decay does not
   * show in the loss function
   * 
   * @return contribution to loss by weight regularization
   * @see NetOptimizers */
  default Scalar reg_loss(boolean decayInLoss) {
    return RealScalar.ZERO;
  }
}
