// code by jph
package ch.alpine.subare.net;

import java.util.Arrays;
import java.util.Random;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.mat.Tolerance;
import ch.alpine.tensor.nrm.Vector2NormSquared;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.sca.Round;

class CNNDemo1Test {
  @Test
  void testSimple() {
    // 1. Data Setup
    double[] input = { 0.0, 0.0, 1.0, 1.0 };
    double[] target = { 0.0, 1.0 }; // The "Step" starts at index 1
    Tensor targ2 = Tensors.vectorDouble(target);
    double learningRate = 0.2;
    // 2. Initialize Conv Layer (Kernel Size 3)
    // A kernel of 3 will look at [0,0,1] then [0,1,1]
    int kernelSize = input.length - target.length + 1;
    Conv1D conv = new Conv1D(kernelSize);
    Conv1DLayer conv1dLayer = Conv1DLayer.of(NormalDistribution.of(0, 0.1), new Random(), kernelSize);
    Scalar totalLoss2 = RealScalar.ONE;
    for (int epoch = 0; epoch <= 100; epoch++) {
      // FORWARD PASS
      double[] pred1 = conv.forward(input);
      Tensor pred2 = conv1dLayer.forward(Tensors.vectorDouble(input));
      // CALCULATE LOSS (MSE) & GRADIENT
      double totalLoss = 0;
      double[] outputGradients = new double[pred1.length];
      for (int i = 0; i < pred1.length; i++) {
        double error = pred1[i] - target[i];
        totalLoss += Math.pow(error, 2);
        // Derivative of (pred - target)^2 is 2 * (pred - target)
        outputGradients[i] = 2 * error;
        // ---
      }
      Tensor outGrad = targ2.subtract(pred2).multiply(RealScalar.TWO);
      totalLoss2 = Vector2NormSquared.of(pred2.subtract(targ2));
      // BACKWARD PASS (Updates weights automatically)
      conv.backward(outputGradients, learningRate);
      Tensor back = conv1dLayer.back(outGrad.multiply(RealScalar.of(learningRate)));
      conv1dLayer.update();
      // PRINT PROGRESS every 20 epochs
      if (epoch % 20 == 0) {
        System.out.printf("Epoch %d | Loss: %.4f | Pred: %s\n", epoch, totalLoss, Arrays.toString(pred1));
        IO.println(" -> " + totalLoss2.maps(Round._3) + " " + pred2.maps(Round._3));
      }
    }
    System.out.println("\nFinal Weights: " + Arrays.toString(conv.weights));
    System.out.println("Final Bias: " + conv.bias);
    IO.println(conv1dLayer.w);
    IO.println(conv1dLayer.b);
    Tolerance.CHOP.requireZero(totalLoss2);
  }
}
