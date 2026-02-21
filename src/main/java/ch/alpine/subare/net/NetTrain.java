// code by jph
package ch.alpine.subare.net;

import java.util.function.Consumer;

import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.qty.Timing;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/NetTrain.html">NetTrain</a> */
public enum NetTrain {
  ;
  /** @param netChain
   * @param xdata input
   * @param ydata output
   * @param learningRate
   * @param consumer
   * @param timeout
   * @param maxEpoch */
  public static void of(NetChain netChain, Tensor xdata, Tensor ydata, Scalar learningRate, Consumer<Tensor> consumer, Scalar timeout, int maxEpoch) {
    int epoch = 0;
    Timing timing = Timing.started();
    while (Scalars.lessThan(timing.seconds(), timeout) && epoch < maxEpoch) {
      for (int i = 0; i < xdata.length(); i++) {
        Tensor y = netChain.forward(xdata.get(i));
        Tensor d = netChain.error(ydata.get(i)).multiply(learningRate);
        netChain.back(d);
        netChain.update();
      }
      if (epoch % 10 == 0)
        consumer.accept(netChain.parameters());
      ++epoch;
    }
  }
}
