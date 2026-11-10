// code by jph
package ch.alpine.subare.net;

import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.alg.Partition;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.nrm.NormalizeTotal;
import ch.alpine.tensor.nrm.Vector2NormSquared;
import ch.alpine.tensor.num.RandomPermutation;
import ch.alpine.tensor.pdf.Permute;
import ch.alpine.tensor.qty.Timing;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.AbsSquared;
import ch.alpine.tensor.sca.exp.Log;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/NetTrain.html">NetTrain</a> */
public class NetTrain {
  private final List<NetTrainListener> list = new LinkedList<>();
  public final TableBuilder tparam = new TableBuilder();
  public final TableBuilder tloss = new TableBuilder();
  private final NetLayer netLayer;
  private final Tensor xdata;
  private final Tensor ydata;

  /** @param netChain
   * @param xdata input
   * @param ydata output */
  public NetTrain(NetLayer netLayer, Tensor xdata, Tensor ydata) {
    this.netLayer = Objects.requireNonNull(netLayer);
    Integers.requireEquals(xdata.length(), ydata.length());
    this.xdata = xdata;
    this.ydata = ydata;
  }

  public void addListener(NetTrainListener runnable) {
    list.add(runnable);
  }

  public void run(Scalar learningRate, Scalar timeout, int maxEpoch, int skip) {
    int epoch = 0;
    Timing timing = Timing.started();
    final int n = xdata.length();
    // 6. Repeat until the loss becomes sufficiently small.
    while (Scalars.lessThan(timing.seconds(), timeout) && epoch < maxEpoch) {
      Tensor lossv = Tensors.reserve(n);
      int[] index = RandomPermutation.of(n, ThreadLocalRandom.current());
      for (int c = 0; c < n; c++) {
        int i = index[c];
        // TODO in would be more desirable to have error = (ypred-ydata)^2
        // 1. Forward propagation: The network generates a prediction.
        netLayer.forward(xdata.get(i));
        // 2. Calculate error: Compare the prediction with the actual output.
        Tensor error = netLayer.error(ydata.get(i));
        // 4. Backpropagation: Calculate gradients of the loss with respect to network weights.
        // TODO grad base on error !
        netLayer.back(error.multiply(learningRate));
        // 5. Weight update: The optimizer adjusts weights to reduce the loss.
        netLayer.update();
        // ---
        // 3. Calculate loss: Apply a loss function to quantify the error.
        Scalar loss = Flatten.scalars(error).map(AbsSquared.FUNCTION).reduce(Scalar::add).orElseThrow();
        lossv.append(loss);
      }
      if (epoch % skip == 0) {
        // TODO plot as LOG instead of compute LOG here
        Scalar loss = Log.FUNCTION.apply(Mean.ofVector(lossv));
        tloss.appendRow(RealScalar.of(epoch), loss);
        tparam.appendRow(RealScalar.of(epoch), Tensor.of(netLayer.parameters()));
        final int fi = epoch;
        list.forEach(ntl -> ntl.epoch(fi));
      }
      ++epoch;
    }
  }

  public void runAsBatches(Scalar learningRate, int batch_size, int maxEpoch, Scalar timeout) {
    int epoch = 0;
    Timing timing = Timing.started();
    final int n = xdata.length();
    // 6. Repeat until the loss becomes sufficiently small.
    while (Scalars.lessThan(timing.seconds(), timeout) && epoch < maxEpoch) {
      Tensor lossv = Tensors.reserve(n);
      int[] index = RandomPermutation.of(n, ThreadLocalRandom.current());
      TensorUnaryOperator tuo = Permute.of(index);
      Tensor batchX = Partition.of(tuo.apply(xdata), batch_size);
      Tensor batchY = Partition.of(tuo.apply(ydata), batch_size);
      for (int i = 0; i < batchX.length(); ++i) {
        // TODO in would be more desirable to have error = (ypred-ydata)^2
        // 1. Forward propagation: The network generates a prediction.
        netLayer.forward(batchX.get(i));
        // 2. Calculate error: Compare the prediction with the actual output.
        Tensor error = netLayer.error(batchY.get(i));
        // 3. Calculate loss: Apply a loss function to quantify the error.
        Tensor relat = Tensor.of(error.stream().map(Vector2NormSquared::of));
        Scalar loss = Mean.ofVector(relat);
        lossv.append(loss);
        Tensor weights = NormalizeTotal.FUNCTION.apply(relat);
        // Flatten.scalars(error).map(AbsSquared.FUNCTION).reduce(Scalar::add).orElseThrow();
        // 4. Backpropagation: Calculate gradients of the loss with respect to network weights.
        // TODO grad base on error !
        Tensor gradOutput = Tensor.of(IntStream.range(0, batch_size) //
            .mapToObj(k -> error.get(k).multiply(weights.Get(k))));
        netLayer.back(gradOutput.multiply(learningRate)); // .multiply(Rational.of(1, batch_size))
        // 5. Weight update: The optimizer adjusts weights to reduce the loss.
        netLayer.update();
        // ---
      }
      {
        // TODO plot as LOG instead of compute LOG here
        Scalar loss = Log.FUNCTION.apply(Mean.ofVector(lossv));
        tloss.appendRow(RealScalar.of(epoch), loss);
        tparam.appendRow(RealScalar.of(epoch), Tensor.of(netLayer.parameters()));
        final int fi = epoch;
        list.forEach(ntl -> ntl.epoch(fi));
      }
      ++epoch;
    }
  }

  @Deprecated
  public Tensor error() {
    Tensor errors = Tensors.reserve(xdata.length());
    for (int sample = 0; sample < xdata.length(); ++sample) {
      Tensor x = xdata.get(sample);
      Tensor y = netLayer.forward(x);
      Tensor e = ydata.get(sample).subtract(y);
      errors.append(e);
    }
    return errors;
  }
}
