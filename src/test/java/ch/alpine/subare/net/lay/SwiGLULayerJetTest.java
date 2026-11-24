package ch.alpine.subare.net.lay;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.jet.JetScalar;
import ch.alpine.tensor.sca.exp.LogisticSigmoid;

class SwiGLULayerJetTest {
  private static Scalar jetOf(double value, double d1) {
    return JetScalar.of(Tensors.vector(value, d1));
  }

  private static double d1(Scalar s) {
    return ((JetScalar) s).vector().Get(1).number().doubleValue();
  }

  /** Evaluates SwiGLU forward pass on a [1, 5, 1] layer for arbitrary Scalar elements. */
  private static Tensor swiGluForward(Tensor xVec, Tensor wGate, Tensor wVal, Tensor wOut) {
    // xVec: [1], wGate: [1, 5], wVal: [1, 5], wOut: [5, 1]
    int hidden = wGate.get(0).length();
    Tensor h = Tensors.reserve(hidden);
    for (int j = 0; j < hidden; ++j) {
      Scalar g = xVec.Get(0).multiply(wGate.Get(0, j));
      Scalar swishG = g.multiply(LogisticSigmoid.FUNCTION.apply(g));
      Scalar v = xVec.Get(0).multiply(wVal.Get(0, j));
      h.append(swishG.multiply(v));
    }
    return Tensors.of(h.dot(wOut.get(Tensor.ALL, 0)));
  }

  @Test
  void testBackAndGradientFalseWithJetScalar() {
    RandomGenerator rng = RandomGeneratorFactory.of("L64X256MixRandom").create(42L);
    SwiGLULayer layer = new SwiGLULayer(1, 5, 1);
    layer.initialize(rng);
    Tensor x = Tensors.vector(0.65);
    Tensor upstream = Tensors.vector(-1.3); // dL / dy for scalar loss L(y) = upstream . y
    // 1. Layer forward + backward
    layer.forward(x);
    Tensor actualGradInput = layer.back(upstream);
    Tensor actualGrads = layer.gradient(false); // [gW_gate, gW_val, gW_out]
    // 2. Check input gradient dL/dx via JetScalar
    Tensor xJet = Tensors.of(jetOf(x.Get(0).number().doubleValue(), 1.0));
    Tensor yJetFromX = swiGluForward(xJet, layer.W_gate, layer.W_val, layer.W_out);
    Scalar lossJetFromX = yJetFromX.Get(0).multiply(upstream.Get(0));
    assertEquals(d1(lossJetFromX), actualGradInput.Get(0).number().doubleValue(), 1e-11);
    // 3. Check weight gradients via JetScalar for each entry of W_gate, W_val, W_out
    Tensor expectedGradGate = Array.zeros(1, 5);
    Tensor expectedGradVal = Array.zeros(1, 5);
    Tensor expectedGradOut = Array.zeros(5, 1);
    for (int j = 0; j < 5; ++j) {
      Tensor wgJet = layer.W_gate.copy();
      wgJet.set(jetOf(layer.W_gate.Get(0, j).number().doubleValue(), 1.0), 0, j);
      double dWg = d1(swiGluForward(x, wgJet, layer.W_val, layer.W_out).Get(0).multiply(upstream.Get(0)));
      expectedGradGate.set(RealScalar.of(dWg), 0, j);
      Tensor wvJet = layer.W_val.copy();
      wvJet.set(jetOf(layer.W_val.Get(0, j).number().doubleValue(), 1.0), 0, j);
      double dWv = d1(swiGluForward(x, layer.W_gate, wvJet, layer.W_out).Get(0).multiply(upstream.Get(0)));
      expectedGradVal.set(RealScalar.of(dWv), 0, j);
      Tensor woJet = layer.W_out.copy();
      woJet.set(jetOf(layer.W_out.Get(j, 0).number().doubleValue(), 1.0), j, 0);
      double dWo = d1(swiGluForward(x, layer.W_gate, layer.W_val, woJet).Get(0).multiply(upstream.Get(0)));
      expectedGradOut.set(RealScalar.of(dWo), j, 0);
    }
    Tensor expectedGrads = Tensors.of(expectedGradGate, expectedGradVal, expectedGradOut);
    for (int m = 0; m < 3; ++m) {
      Tensor diff = actualGrads.get(m).subtract(expectedGrads.get(m));
      double maxAbsErr = Flatten.scalars(diff) //
          .map(Scalar.class::cast) //
          .mapToDouble(s -> Math.abs(s.number().doubleValue())) //
          .max() //
          .orElse(0.0);
      assertEquals(0.0, maxAbsErr, 1e-11);
    }
  }
}
