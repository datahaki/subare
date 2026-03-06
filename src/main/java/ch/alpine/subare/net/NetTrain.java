// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.qty.Timing;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/NetTrain.html">NetTrain</a> */
public class NetTrain {
  /** @param netChain
   * @param xdata input
   * @param ydata output
   * @param learningRate
   * @param consumer
   * @param timeout
   * @param maxEpoch */
  public final TableBuilder tparam = new TableBuilder();
  // TODO check terminology
  public final TableBuilder tloss = new TableBuilder();
  private final NetChain netChain;
  private final Tensor xdata;
  private final Tensor ydata;

  public NetTrain(NetChain netChain, Tensor xdata, Tensor ydata) {
    Integers.requireEquals(xdata.length(), ydata.length());
    this.netChain = netChain;
    this.xdata = xdata;
    this.ydata = ydata;
  }

  public void run(Scalar learningRate, Scalar timeout, int maxEpoch, int skip) {
    int epoch = 0;
    Timing timing = Timing.started();
    while (Scalars.lessThan(timing.seconds(), timeout) && epoch < maxEpoch) {
      Tensor ds = Tensors.reserve(xdata.length());
      for (int i = 0; i < xdata.length(); i++) {
        Tensor y = netChain.forward(xdata.get(i));
        Tensor d = netChain.error(ydata.get(i)).multiply(learningRate);
        // IO.println("error: "+d);
        ds.append(d);
        netChain.back(d);
        netChain.update();
      }
      if (epoch % skip == 0) {
        tloss.appendRow(RealScalar.of(epoch), ds);
        tparam.appendRow(RealScalar.of(epoch), netChain.parameters());
      }
      ++epoch;
    }
  }

  public Tensor error() {
    Tensor errors = Tensors.reserve(xdata.length());
    for (int sample = 0; sample < xdata.length(); ++sample) {
      Tensor x = xdata.get(sample);
      Tensor y = netChain.forward(x);
      Tensor e = ydata.get(sample).subtract(y);
      errors.append(e);
    }
    return errors;
  }
}
