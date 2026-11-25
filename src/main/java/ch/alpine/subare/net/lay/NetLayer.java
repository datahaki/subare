// code by jph
package ch.alpine.subare.net.lay;

import java.util.random.RandomGenerator;
import java.util.stream.Stream;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/** Inference is running a trained model in production (or on new live/unseen
 * inputs) to generate predictions or outputs for real tasks.
 * 
 * Evaluation is running a trained model on a held-out benchmark or test
 * dataset to measure how well it generalizes and calculate metrics (such
 * as accuracy, precision, recall, F1, or loss) before or during deployment. */
public interface NetLayer {
  /** @param x
   * @return forward of x */
  Tensor forward(Tensor x);

  /** @param X
   * @return forward of entire batch */
  Tensor forwardBatch(Tensor X);

  /** @param gradOutput
   * @return */
  Tensor back(Tensor gradOutput);

  /** back propagation of gradient of entire batch
   * 
   * @param gradOutput
   * @return */
  Tensor backBatch(Tensor gradOutput);

  /** visits all members
   * useful for pointwise operations such as
   * {@link #initialize(RandomGenerator)}
   * {@link #parameters()}
   * {@link #gradient(boolean)}
   * {@link #increment(boolean, Tensor, Scalar)}
   * {@link #reg_loss(boolean)}
   * 
   * @return all net layers directly harbored by this instance */
  default Stream<NetLayer> nested() {
    return Stream.empty();
  }
}
