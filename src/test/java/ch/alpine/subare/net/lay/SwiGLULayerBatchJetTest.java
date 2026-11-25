package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.reg.NoRegularization;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.jet.JetScalar;
import ch.alpine.tensor.sca.exp.LogisticSigmoid;

class SwiGLULayerBatchJetTest {
  private static final int D_IN = 2;
  private static final int D_HIDDEN = 4;
  private static final int D_OUT = 3;
  private static final int BATCH = 2;

  private static Scalar jetOf(double value, double d1) {
    return JetScalar.of(Tensors.vector(value, d1));
  }

  private static double d1(Scalar s) {
    return ((JetScalar) s).jet().Get(1).number().doubleValue();
  }

  /** Converts a Tensor of RealScalars into a matching Tensor of JetScalars with d1 = 0. */
  private static Tensor toZeroJetTensor(Tensor t) {
    return t.maps(s -> jetOf(s.number().doubleValue(), 0.0));
  }

  /** Forward pass for one sample xVec [D_IN] -> yVec [D_OUT],
   * using arbitrary Scalar elements (RealScalar or JetScalar). */
  private static Tensor swiGluSingleForward(Tensor xVec, Tensor wGate, Tensor wVal, Tensor wOut) {
    // g = xVec . wGate -> [D_HIDDEN]
    // v = xVec . wVal -> [D_HIDDEN]
    Tensor g = xVec.dot(wGate);
    Tensor v = xVec.dot(wVal);
    Tensor h = Tensors.reserve(D_HIDDEN);
    for (int j = 0; j < D_HIDDEN; ++j) {
      Scalar gj = g.Get(j);
      Scalar swishGj = gj.multiply(LogisticSigmoid.FUNCTION.apply(gj));
      h.append(swishGj.multiply(v.Get(j)));
    }
    // yVec [D_OUT]: entry k is h . wOut[:, k]
    Tensor yVec = Tensors.reserve(D_OUT);
    for (int k = 0; k < D_OUT; ++k) {
      yVec.append(h.dot(wOut.get(Tensor.ALL, k)));
    }
    return yVec;
  }

  /** Scalar batch loss L = sum_{b, k} upstream[b, k] * yBatch[b, k]. */
  private static Scalar batchLoss(Tensor xBatch, Tensor wGate, Tensor wVal, Tensor wOut, Tensor upstreamBatch) {
    Scalar total = null;
    for (int b = 0; b < BATCH; ++b) {
      Tensor yRow = swiGluSingleForward(xBatch.get(b), wGate, wVal, wOut);
      for (int k = 0; k < D_OUT; ++k) {
        Scalar term = yRow.Get(k).multiply(upstreamBatch.Get(b, k));
        total = (total == null) ? term : total.add(term);
      }
    }
    return total;
  }

  private static double maxAbsDiff(Tensor a, Tensor b) {
    return Flatten.scalars(a.subtract(b)).map(Scalar.class::cast).mapToDouble(s -> Math.abs(s.number().doubleValue())).max().orElse(0.0);
  }

  @Test
  void testBatchBackAndGradientTrueWithJetScalar() {
    RandomGenerator rng = RandomGeneratorFactory.of("L64X256MixRandom").create(7L);
    SwiGLULayer layer = new SwiGLULayer(D_IN, D_HIDDEN, D_OUT);
    layer.netRegularization = NoRegularization.INSTANCE;
    layer.initialize(rng);
    Tensor xBatch = Tensors.matrixDouble(new double[][] { { 0.5, -0.3 }, { 0.0, 0.8 } });
    Tensor upstreamBatch = Tensors.matrixDouble(new double[][] { { 1.0, -0.5, 0.2 }, { -0.4, 0.7, -1.1 } });
    // 1. Layer forwardBatch + backward
    layer.forwardBatch(xBatch);
    Tensor actualGradInputBatch = layer.backBatch(upstreamBatch);
    Tensor actualGrads = layer.gradient(true); // [gW_gate, gW_val, gW_out]
    Tensor wGateZeroJet = toZeroJetTensor(layer.W_gate);
    Tensor wValZeroJet = toZeroJetTensor(layer.W_val);
    Tensor wOutZeroJet = toZeroJetTensor(layer.W_out);
    Tensor upstreamZeroJet = toZeroJetTensor(upstreamBatch);
    // 2. Check input gradients dL / dxBatch[b, i]
    Tensor expectedGradInputBatch = Array.zeros(BATCH, D_IN);
    for (int b = 0; b < BATCH; ++b) {
      for (int i = 0; i < D_IN; ++i) {
        Tensor xJetBatch = toZeroJetTensor(xBatch);
        xJetBatch.set(jetOf(xBatch.Get(b, i).number().doubleValue(), 1.0), b, i);
        Scalar lossJet = batchLoss(xJetBatch, wGateZeroJet, wValZeroJet, wOutZeroJet, upstreamZeroJet);
        expectedGradInputBatch.set(RealScalar.of(d1(lossJet)), b, i);
      }
    }
    assertEquals(0.0, maxAbsDiff(actualGradInputBatch, expectedGradInputBatch), 1e-11);
    // 3. Check W_gate gradients [D_IN, D_HIDDEN]
    Tensor expectedGradGate = Array.zeros(D_IN, D_HIDDEN);
    Tensor xZeroJetBatch = toZeroJetTensor(xBatch);
    for (int i = 0; i < D_IN; ++i) {
      for (int j = 0; j < D_HIDDEN; ++j) {
        Tensor wgJet = toZeroJetTensor(layer.W_gate);
        wgJet.set(jetOf(layer.W_gate.Get(i, j).number().doubleValue(), 1.0), i, j);
        Scalar lossJet = batchLoss(xZeroJetBatch, wgJet, wValZeroJet, wOutZeroJet, upstreamZeroJet);
        expectedGradGate.set(RealScalar.of(d1(lossJet)), i, j);
      }
    }
    IO.println("actualGrads.get(0)" + actualGrads.get(0));
    IO.println("expectedGradGate=" + expectedGradGate);
    assertEquals(0.0, maxAbsDiff(actualGrads.get(0), expectedGradGate), 1e-11);
    // 4. Check W_val gradients [D_IN, D_HIDDEN]
    Tensor expectedGradVal = Array.zeros(D_IN, D_HIDDEN);
    for (int i = 0; i < D_IN; ++i) {
      for (int j = 0; j < D_HIDDEN; ++j) {
        Tensor wvJet = toZeroJetTensor(layer.W_val);
        wvJet.set(jetOf(layer.W_val.Get(i, j).number().doubleValue(), 1.0), i, j);
        Scalar lossJet = batchLoss(xZeroJetBatch, wGateZeroJet, wvJet, wOutZeroJet, upstreamZeroJet);
        expectedGradVal.set(RealScalar.of(d1(lossJet)), i, j);
      }
    }
    assertEquals(0.0, maxAbsDiff(actualGrads.get(1), expectedGradVal), 1e-11);
    // 5. Check W_out gradients [D_HIDDEN, D_OUT]
    Tensor expectedGradOut = Array.zeros(D_HIDDEN, D_OUT);
    for (int j = 0; j < D_HIDDEN; ++j) {
      for (int k = 0; k < D_OUT; ++k) {
        Tensor woJet = toZeroJetTensor(layer.W_out);
        woJet.set(jetOf(layer.W_out.Get(j, k).number().doubleValue(), 1.0), j, k);
        Scalar lossJet = batchLoss(xZeroJetBatch, wGateZeroJet, wValZeroJet, woJet, upstreamZeroJet);
        expectedGradOut.set(RealScalar.of(d1(lossJet)), j, k);
      }
    }
    assertEquals(0.0, maxAbsDiff(actualGrads.get(2), expectedGradOut), 1e-11);
  }
}
