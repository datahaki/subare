// code by jph
package ch.alpine.subare.net;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Partition;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.chq.DeterminateTensorQ;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.pdf.Permute;
import ch.alpine.tensor.qty.Timing;
import ch.alpine.tensor.red.Total;
import ch.alpine.tensor.sca.exp.Log;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/NetTrain.html">NetTrain</a> */
public class NetTrain implements Serializable {
  private final transient List<NetTrainListener> list = new LinkedList<>();
  public final NetLayer netLayer;
  private final transient TrainingData trainingData;
  public final TableBuilder tloss = new TableBuilder();

  /** @param netChain
   * @param xdata input
   * @param ydata output */
  public NetTrain(NetLayer netLayer, TrainingData trainingData) {
    this.netLayer = Objects.requireNonNull(netLayer);
    this.trainingData = trainingData;
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
   * @param max_epochs
   * @param timeout */
  public void runAsBatches(Scalar learningRate, int batch_size, int max_epochs, Scalar timeout) {
    int epoch = 0;
    Timing timing = Timing.started();
    Tensor xdata = trainingData.xdata();
    Tensor ydata = trainingData.ydata();
    final int n = xdata.length();
    // 6. Repeat until the loss becomes sufficiently small.
    while (Scalars.lessThan(timing.seconds(), timeout) && epoch <= max_epochs) {
      // 0. Shuffle dataset indices at the beginning of every epoch
      // (Crucial for preventing the model from memorizing batch order)
      TensorUnaryOperator tuo = Permute.range(n);
      Tensor batchX = Partition.gather(tuo.apply(xdata), batch_size);
      Tensor batchY = Partition.gather(tuo.apply(ydata), batch_size);
      Tensor lossv = Tensors.reserve(batchX.length()); // to compute epoch_loss
      for (int i = 0; i < batchX.length(); ++i) {
        Tensor x = batchX.get(i);
        int samples = x.length();
        // 1. Forward propagation: The network generates a prediction.
        Tensor prediction = netLayer.forward(x);
        DeterminateTensorQ.require(prediction);
        // 2. Calculate error: Compare the prediction with the actual output.
        Tensor target = batchY.get(i);
        // IO.println("T="+Dimensions.of(target));
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
        final int fi = epoch;
        boolean cont = list.stream() //
            .map(ntl -> ntl.epoch(this, fi)) //
            .reduce(Boolean::logicalAnd) //
            .orElse(true);
        if (!cont)
          break;
      }
      ++epoch;
    }
  }

  /** @return log of loss value at final training step */
  public Scalar log_loss() {
    return tloss.getRow(tloss.getRowCount() - 1).Get(1);
  }
}
