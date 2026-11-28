// code by jph
package ch.alpine.subare.net.run;

import java.util.LinkedList;
import java.util.List;

import ch.alpine.subare.net.lay.DropoutLayers;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.subare.net.los.LossFunction;
import ch.alpine.subare.net.los.NetLoss;
import ch.alpine.subare.net.mod.NetModel;
import ch.alpine.subare.net.opt.NetOptimizer;
import ch.alpine.subare.net.opt.NetOptimizerFactory;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Partition;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.chq.DeterminateTensorQ;
import ch.alpine.tensor.io.TableBuilder;
import ch.alpine.tensor.pdf.Permute;
import ch.alpine.tensor.qty.Timing;
import ch.alpine.tensor.red.MeanSummaryStatistics;
import ch.alpine.tensor.red.Total;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/NetTrain.html">NetTrain</a> */
public class NetTrain {
  public static final String[] LOSS_LABELS = { "data", "regularization" };
  private final transient List<NetTrainListener> list = new LinkedList<>();
  private final NetModel netModel;
  private final NetOptimizer netOptimizer;
  private final MeanSummaryStatistics mss = new MeanSummaryStatistics();
  public final TableBuilder tloss = new TableBuilder();

  /** @param netChain
   * @param trainingData output */
  public NetTrain(NetModel netModel, NetOptimizerFactory netLayerOptimizer) {
    this.netModel = netModel;
    this.netOptimizer = netLayerOptimizer.apply(netModel.netLayer());
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
  public void runAsBatches(TrainingData trainingData, int batch_size) {
    NetLayer netLayer = netModel.netLayer();
    LossFunction lossFunction = netModel.lossFunction();
    DropoutLayers.setEnabled(netLayer, true);
    // NetLayers.toString(netLayer).forEach(IO::println);
    Tensor xdata = trainingData.xdata();
    Tensor ydata = trainingData.ydata();
    final int n = xdata.length();
    if (n == 0) {
      System.err.println("trainingData is empty");
      return;
    }
    {
      int rem = n % batch_size;
      if (rem != 0)
        System.err.println("warning batch_size: " + rem + " != " + batch_size);
    }
    // 6. Repeat until the loss becomes sufficiently small.
    boolean allMatch = true;
    while (allMatch) {
      Timing timing = Timing.started();
      // 0. Shuffle dataset indices at the beginning of every epoch
      // (Crucial for preventing the model from memorizing batch order)
      TensorUnaryOperator permute = Permute.range(n);
      Tensor batchX = Partition.gather(permute.apply(xdata), batch_size);
      Tensor batchY = Partition.gather(permute.apply(ydata), batch_size);
      Tensor lossv = Tensors.reserve(batchX.length()); // to compute epoch_loss
      for (int i = 0; i < batchX.length(); ++i) {
        Tensor x = batchX.get(i);
        int samples = x.length();
        // 1. Forward propagation: The network generates a prediction.
        Tensor prediction = netLayer.forwardBatch(x);
        DeterminateTensorQ.require(prediction);
        // 2. Calculate error: Compare the prediction with the actual output.
        Tensor target = batchY.get(i);
        // IO.println("T="+Dimensions.of(target));
        NetLoss netError = lossFunction.computeBatch(prediction, target);
        // 3. Calculate loss: Apply a loss function to quantify the error.
        /** Loss:
         * A scalar value computed by a loss function (e.g., Mean
         * Squared Error, Cross-Entropy) that aggregates the errors
         * across a sample or batch. It quantifies overall model
         * performance and serves as the objective function to
         * minimize. */
        lossv.append(netError.loss().multiply(RealScalar.of(samples)));
        // 4. Backpropagation: Calculate gradients of the loss with respect to network weights.
        netLayer.backBatch(netError.gradOutput());
        // 5. Weight update: The optimizer adjusts weights to reduce the loss.
        netOptimizer.update();
      }
      int epoch = tloss.getRowCount();
      Scalar loss1 = Total.ofVector(lossv).divide(RealScalar.of(n));
      Scalar loss2 = NetLayers.reg_loss(netLayer, netOptimizer.weightDecay());
      if (0 < epoch) {
        mss.accept(timing.seconds());
        IO.println("mean=" + mss.getMean().orElseThrow());
      }
      tloss.appendRow(RealScalar.of(epoch), loss1, loss2);
      allMatch &= list.stream() //
          .map(ntl -> ntl.epoch(this)) //
          .reduce(Boolean::logicalAnd) //
          .orElseThrow();
    }
    DropoutLayers.setEnabled(netLayer, false);
  }

  public int getTotalEpochs() {
    return tloss.getRowCount();
  }

  public void addListener(NetTrainListener runnable) {
    list.add(runnable);
  }

  public void setMaxEpochs(int max_epochs) {
    addListener(new MaxEpochListener(max_epochs));
  }

  /** @return */
  public NetModel netModel() {
    return netModel;
  }

  public NetOptimizer netOptimizer() {
    return netOptimizer;
  }
}
