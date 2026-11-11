// code by jph
package ch.alpine.subare.net;

import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Partition;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.num.RandomPermutation;
import ch.alpine.tensor.pdf.Permute;
import ch.alpine.tensor.qty.Timing;
import ch.alpine.tensor.red.Total;
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

  /** Epoch: One complete pass through the entire training dataset. An
   * epoch consists of ⌈N /B⌉ iterations (or steps), where N is
   * total dataset size.
   * 
   * @param learningRate A scalar hyperparameter that scales the magnitude of the
   * weight update during optimization (W ← W − η ⋅ ∇W L).
   * 
   * @param batch_size The number of independent samples processed together in
   * a single forward and backward pass. Enables parallel matrix
   * operations on hardware (GPUs/CPUs).
   * 
   * @param maxEpoch
   * @param timeout */
  public void runAsBatches(Scalar learningRate, int batch_size, int maxEpoch, Scalar timeout) {
    int epoch = 0;
    Timing timing = Timing.started();
    final int n = xdata.length();
    // 6. Repeat until the loss becomes sufficiently small.
    while (Scalars.lessThan(timing.seconds(), timeout) && epoch <= maxEpoch) {
      // 0. Shuffle dataset indices at the beginning of every epoch
      // (Crucial for preventing the model from memorizing batch order)
      int[] index = RandomPermutation.of(n, ThreadLocalRandom.current());
      TensorUnaryOperator tuo = Permute.of(index);
      Tensor batchX = Partition.gather(tuo.apply(xdata), batch_size);
      Tensor batchY = Partition.gather(tuo.apply(ydata), batch_size);
      Tensor lossv = Tensors.reserve(batchX.length()); // to compute epoch_loss
      for (int i = 0; i < batchX.length(); ++i) {
        Tensor x = batchX.get(i);
        int samples = x.length();
        // 1. Forward propagation: The network generates a prediction.
        Tensor prediction = netLayer.forward(x);
        // 2. Calculate error: Compare the prediction with the actual output.
        Tensor target = batchY.get(i);
        NetError netError = netLayer.error(target);
        // 3. Calculate loss: Apply a loss function to quantify the error.
        /** Loss:
         * A scalar value computed by a loss function (e.g., Mean
         * Squared Error, Cross-Entropy) that aggregates the errors
         * across a sample or batch. It quantifies overall model
         * performance and serves as the objective function to
         * minimize. */
        lossv.append(netError.loss().multiply(RealScalar.of(samples)));
        // 4. Backpropagation: Calculate gradients of the loss with respect to network weights.
        netLayer.back(netError.gradOutput());
        // 5. Weight update: The optimizer adjusts weights to reduce the loss.
        netLayer.update(learningRate);
      }
      {
        // TODO plot as LOG instead of compute LOG here
        Scalar loss = Log.FUNCTION.apply(Total.ofVector(lossv).divide(RealScalar.of(xdata.length())));
        tloss.appendRow(RealScalar.of(epoch), loss);
        tparam.appendRow(RealScalar.of(epoch), Tensor.of(netLayer.parameters()));
        final int fi = epoch;
        list.forEach(ntl -> ntl.epoch(fi));
      }
      ++epoch;
    }
  }

  /** @return log of loss value at final training step */
  public Scalar log_loss() {
    return tloss.getRow(tloss.getRowCount() - 1).Get(1);
  }
}
