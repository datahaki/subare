// code by gemini
package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.opt.NetOptimizer;
import ch.alpine.subare.net.opt.NetOptimizers;
import ch.alpine.subare.net.reg.L2Regularization;
import ch.alpine.subare.net.reg.NoRegularization;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.ConstantArray;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.mat.HilbertMatrix;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.red.Max;
import ch.alpine.tensor.sca.Abs;

class SwiGLULayerTest {
  @Test
  void test() {
    SwiGLULayer swiGLULayer = new SwiGLULayer(3, 10, 6);
    NetLayers.initialize(swiGLULayer);
    {
      Tensor y = swiGLULayer.forward(Tensors.vector(1, 2, 3));
      VectorQ.requireLength(y, 6);
      Tensor z = swiGLULayer.back(Tensors.vector(1, 2, 3, 2, 0, -1));
      VectorQ.requireLength(z, 3);
    }
    NetOptimizer netOptimizer = NetOptimizers.ADAM_W.create(swiGLULayer, RealScalar.of(0.03), RealScalar.of(0.9), RealScalar.of(0.99));
    {
      Tensor Y = swiGLULayer.forwardBatch(HilbertMatrix.of(10, 3));
      Tensor Z = swiGLULayer.backBatch(Y);
      Z.maps(Scalar::zero);
    }
    netOptimizer.update();
  }

  private static final double EPS = 1e-6;
  private static final double TOL = 1e-5;

  private static Tensor scalarLoss(Tensor output, Tensor upstream) {
    // Dot product <output, upstream> gives scalar loss with gradient = upstream
    return output.dot(upstream);
  }

  private static Tensor batchScalarLoss(Tensor outputBatch, Tensor upstreamBatch) {
    Scalar sum = RealScalar.ZERO;
    for (int b = 0; b < outputBatch.length(); ++b) {
      for (int j = 0; j < outputBatch.get(b).length(); ++j) {
        sum = sum.add(outputBatch.Get(b, j).multiply(upstreamBatch.Get(b, j)));
      }
    }
    return sum;
  }

  private static Tensor numericalGradTensor(Tensor target, java.util.function.UnaryOperator<Tensor> evalScalarLoss) {
    List<Integer> dims = Dimensions.of(target);
    if (dims.size() == 1) {
      return Tensors.vector(i -> centralDiff1D(target, i, evalScalarLoss), dims.get(0));
    }
    if (dims.size() == 2) {
      return Tensor
          .of(java.util.stream.IntStream.range(0, dims.get(0)).mapToObj(r -> Tensors.vector(c -> centralDiff2D(target, r, c, evalScalarLoss), dims.get(1))));
    }
    throw new IllegalArgumentException("Unsupported rank: " + dims.size());
  }

  private static Scalar centralDiff1D(Tensor tensor, int i, java.util.function.UnaryOperator<Tensor> evalScalarLoss) {
    Tensor copy = tensor.copy();
    Scalar orig = copy.Get(i);
    copy.set(orig.add(RealScalar.of(EPS)), i);
    Scalar plus = (Scalar) evalScalarLoss.apply(copy);
    copy.set(orig.subtract(RealScalar.of(EPS)), i);
    Scalar minus = (Scalar) evalScalarLoss.apply(copy);
    copy.set(orig, i);
    return plus.subtract(minus).divide(RealScalar.of(2.0 * EPS));
  }

  private static Scalar centralDiff2D(Tensor tensor, int r, int c, java.util.function.UnaryOperator<Tensor> evalScalarLoss) {
    Tensor copy = tensor.copy();
    Scalar orig = copy.Get(r, c);
    copy.set(orig.add(RealScalar.of(EPS)), r, c);
    Scalar plus = (Scalar) evalScalarLoss.apply(copy);
    copy.set(orig.subtract(RealScalar.of(EPS)), r, c);
    Scalar minus = (Scalar) evalScalarLoss.apply(copy);
    copy.set(orig, r, c);
    return plus.subtract(minus).divide(RealScalar.of(2.0 * EPS));
  }

  private static void assertTensorClose(Tensor actual, Tensor expected, String label) {
    Tensor diff = actual.subtract(expected).maps(Abs.FUNCTION);
    Scalar maxDiff = Flatten.scalars(diff).reduce(Max::of).orElseThrow();
    // diff.flatten(-1).map(Scalar.class::cast).reduce(RealScalar.ZERO, (a, b) -> Scalars.lessThan(a, b) ? b : a);
    assertEquals(true, Scalars.lessThan(maxDiff, RealScalar.of(TOL)), label + " max finite-diff error too large: " + maxDiff.number().doubleValue());
  }

  private static SwiGLULayer newInitializedLayer(RandomGenerator rng, int dIn, int dHidden, int dOut) {
    SwiGLULayer layer = new SwiGLULayer(dIn, dHidden, dOut);
    layer.initialize(rng);
    Distribution dist = NormalDistribution.of(0.0, 0.5);
    layer.W_gate = RandomVariate.of(dist, rng, dIn, dHidden);
    layer.W_val = RandomVariate.of(dist, rng, dIn, dHidden);
    layer.W_out = RandomVariate.of(dist, rng, dHidden, dOut);
    return layer;
  }

  @Test
  void testBackSingleSample() {
    RandomGenerator rng = RandomGeneratorFactory.of("L64X256MixRandom").create(42L);
    int dIn = 4, dHidden = 6, dOut = 3;
    SwiGLULayer layer = newInitializedLayer(rng, dIn, dHidden, dOut);
    layer.netRegularization = NoRegularization.INSTANCE;
    Distribution dist = NormalDistribution.of(0.0, 0.5);
    Tensor x = RandomVariate.of(dist, rng, dIn);
    Tensor gradOut = RandomVariate.of(dist, rng, dOut);
    layer.forward(x);
    Tensor actualGradX = layer.back(gradOut);
    Tensor actualGrads = layer.gradient(true);
    Tensor numGradX = numericalGradTensor(x, xPerturbed -> {
      SwiGLULayer probe = new SwiGLULayer(dIn, dHidden, dOut);
      probe.W_gate = layer.W_gate.copy();
      probe.W_val = layer.W_val.copy();
      probe.W_out = layer.W_out.copy();
      return scalarLoss(probe.forward(xPerturbed), gradOut);
    });
    Tensor numGradWGate = numericalGradTensor(layer.W_gate, wPerturbed -> {
      SwiGLULayer probe = new SwiGLULayer(dIn, dHidden, dOut);
      probe.W_gate = wPerturbed;
      probe.W_val = layer.W_val.copy();
      probe.W_out = layer.W_out.copy();
      return scalarLoss(probe.forward(x), gradOut);
    });
    Tensor numGradWVal = numericalGradTensor(layer.W_val, wPerturbed -> {
      SwiGLULayer probe = new SwiGLULayer(dIn, dHidden, dOut);
      probe.W_gate = layer.W_gate.copy();
      probe.W_val = wPerturbed;
      probe.W_out = layer.W_out.copy();
      return scalarLoss(probe.forward(x), gradOut);
    });
    Tensor numGradWOut = numericalGradTensor(layer.W_out, wPerturbed -> {
      SwiGLULayer probe = new SwiGLULayer(dIn, dHidden, dOut);
      probe.W_gate = layer.W_gate.copy();
      probe.W_val = layer.W_val.copy();
      probe.W_out = wPerturbed;
      return scalarLoss(probe.forward(x), gradOut);
    });
    assertTensorClose(actualGradX, numGradX, "back: gradInput");
    assertTensorClose(actualGrads.get(0), numGradWGate, "back: gradW_gate");
    assertTensorClose(actualGrads.get(1), numGradWVal, "back: gradW_val");
    assertTensorClose(actualGrads.get(2), numGradWOut, "back: gradW_out");
  }

  @Test
  void testBackBatch() {
    RandomGenerator rng = RandomGeneratorFactory.of("L64X256MixRandom").create(123L);
    int batchSize = 5, dIn = 4, dHidden = 6, dOut = 3;
    SwiGLULayer layer = newInitializedLayer(rng, dIn, dHidden, dOut);
    layer.netRegularization = NoRegularization.INSTANCE;
    Distribution dist = NormalDistribution.of(0.0, 0.5);
    Tensor X = RandomVariate.of(dist, rng, batchSize, dIn);
    Tensor gradOutBatch = RandomVariate.of(dist, rng, batchSize, dOut);
    layer.forwardBatch(X);
    Tensor actualGradX = layer.backBatch(gradOutBatch);
    Tensor actualGrads = layer.gradient(true);
    Tensor numGradX = numericalGradTensor(X, xPerturbed -> {
      SwiGLULayer probe = new SwiGLULayer(dIn, dHidden, dOut);
      probe.W_gate = layer.W_gate.copy();
      probe.W_val = layer.W_val.copy();
      probe.W_out = layer.W_out.copy();
      return batchScalarLoss(probe.forwardBatch(X), gradOutBatch);
    });
    // Re-run with xPerturbed in the lambda capture:
    numGradX = numericalGradTensor(X, xPerturbed -> {
      SwiGLULayer probe = new SwiGLULayer(dIn, dHidden, dOut);
      probe.W_gate = layer.W_gate.copy();
      probe.W_val = layer.W_val.copy();
      probe.W_out = layer.W_out.copy();
      return batchScalarLoss(probe.forwardBatch(xPerturbed), gradOutBatch);
    });
    Tensor numGradWGate = numericalGradTensor(layer.W_gate, wPerturbed -> {
      SwiGLULayer probe = new SwiGLULayer(dIn, dHidden, dOut);
      probe.W_gate = wPerturbed;
      probe.W_val = layer.W_val.copy();
      probe.W_out = layer.W_out.copy();
      return batchScalarLoss(probe.forwardBatch(X), gradOutBatch);
    });
    Tensor numGradWVal = numericalGradTensor(layer.W_val, wPerturbed -> {
      SwiGLULayer probe = new SwiGLULayer(dIn, dHidden, dOut);
      probe.W_gate = layer.W_gate.copy();
      probe.W_val = wPerturbed;
      probe.W_out = layer.W_out.copy();
      return batchScalarLoss(probe.forwardBatch(X), gradOutBatch);
    });
    Tensor numGradWOut = numericalGradTensor(layer.W_out, wPerturbed -> {
      SwiGLULayer probe = new SwiGLULayer(dIn, dHidden, dOut);
      probe.W_gate = layer.W_gate.copy();
      probe.W_val = layer.W_val.copy();
      probe.W_out = wPerturbed;
      return batchScalarLoss(probe.forwardBatch(X), gradOutBatch);
    });
    assertTensorClose(actualGradX, numGradX, "backBatch: gradInput");
    assertTensorClose(actualGrads.get(0), numGradWGate, "backBatch: gradW_gate");
    assertTensorClose(actualGrads.get(1), numGradWVal, "backBatch: gradW_val");
    assertTensorClose(actualGrads.get(2), numGradWOut, "backBatch: gradW_out");
  }

  @Test
  void testForwardMatchesForwardBatchRowByRow() {
    RandomGenerator rng = RandomGeneratorFactory.of("L64X256MixRandom").create(77L);
    int batchSize = 5, dIn = 4, dHidden = 6, dOut = 3;
    SwiGLULayer layer = new SwiGLULayer(dIn, dHidden, dOut);
    layer.initialize(rng);
    Distribution dist = NormalDistribution.of(0.0, 0.5);
    Tensor X = RandomVariate.of(dist, rng, batchSize, dIn);
    Tensor batchOut = layer.forwardBatch(X);
    assertEquals(batchSize, batchOut.length());
    for (int b = 0; b < batchSize; ++b) {
      Tensor singleOut = layer.forward(X.get(b));
      assertTensorClose(singleOut, batchOut.get(b), "row " + b + " forward vs forwardBatch");
    }
  }

  @Test
  void testBackMatchesBackBatchRowByRow() {
    RandomGenerator rng = RandomGeneratorFactory.of("L64X256MixRandom").create(99L);
    int batchSize = 5, dIn = 4, dHidden = 6, dOut = 3;
    SwiGLULayer layer = new SwiGLULayer(dIn, dHidden, dOut);
    layer.netRegularization = new L2Regularization(RealScalar.ZERO);
    layer.initialize(rng);
    Distribution dist = NormalDistribution.of(0.0, 0.5);
    Tensor X = RandomVariate.of(dist, rng, batchSize, dIn);
    Tensor gradOutBatch = RandomVariate.of(dist, rng, batchSize, dOut);
    layer.forwardBatch(X);
    Tensor batchGradX = layer.backBatch(gradOutBatch);
    Tensor batchGrads = layer.gradient(true);
    assertEquals(batchSize, batchGradX.length());
    Tensor accumulatedWGate = ConstantArray.of(RealScalar.ZERO, dIn, dHidden);
    Tensor accumulatedWVal = ConstantArray.of(RealScalar.ZERO, dIn, dHidden);
    Tensor accumulatedWOut = ConstantArray.of(RealScalar.ZERO, dHidden, dOut);
    for (int b = 0; b < batchSize; ++b) {
      layer.forward(X.get(b));
      Tensor singleGradX = layer.back(gradOutBatch.get(b));
      assertTensorClose(singleGradX, batchGradX.get(b), "row " + b + " back vs backBatch gradInput");
      Tensor rowGrads = layer.gradient(true);
      accumulatedWGate = accumulatedWGate.add(rowGrads.get(0));
      accumulatedWVal = accumulatedWVal.add(rowGrads.get(1));
      accumulatedWOut = accumulatedWOut.add(rowGrads.get(2));
    }
    assertTensorClose(accumulatedWGate, batchGrads.get(0), "sum of back vs backBatch gradW_gate");
    assertTensorClose(accumulatedWVal, batchGrads.get(1), "sum of back vs backBatch gradW_val");
    assertTensorClose(accumulatedWOut, batchGrads.get(2), "sum of back vs backBatch gradW_out");
  }
}
