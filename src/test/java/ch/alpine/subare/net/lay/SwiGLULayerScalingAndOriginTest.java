package ch.alpine.subare.net.lay;

import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.net.reg.NoRegularization;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.mat.Tolerance;

class SwiGLULayerScalingAndOriginTest {
  @Test
  void testZeroInputOriginBehavior() {
    RandomGenerator rng = RandomGeneratorFactory.of("L64X256MixRandom").create(17L);
    SwiGLULayer layer = new SwiGLULayer(2, 4, 3);
    layer.netRegularization = NoRegularization.INSTANCE;
    layer.initialize(rng);
    Tensor zeroX = Array.zeros(2);
    Tensor gradOut = Tensors.vector(0.5, -1.2, 0.3);
    Tensor y = layer.forward(zeroX);
    Tolerance.CHOP.requireClose(y, Array.zeros(3));
    layer.back(gradOut);
    Tensor grads = layer.gradient(false);
    Tolerance.CHOP.requireClose(grads.get(0), Array.zeros(2, 4)); // gW_gate
    Tolerance.CHOP.requireClose(grads.get(1), Array.zeros(2, 4)); // gW_val
    Tolerance.CHOP.requireClose(grads.get(2), Array.zeros(4, 3)); // gW_out
  }

  @Test
  void testValueAndOutputScalingInvariance() {
    RandomGenerator rng = RandomGeneratorFactory.of("L64X256MixRandom").create(23L);
    Scalar c = RealScalar.of(2.5);
    SwiGLULayer base = new SwiGLULayer(2, 4, 3);
    base.netRegularization = NoRegularization.INSTANCE;
    base.initialize(rng);
    SwiGLULayer scaled = new SwiGLULayer(2, 4, 3);
    scaled.netRegularization = NoRegularization.INSTANCE;
    scaled.W_gate = base.W_gate.copy();
    scaled.W_val = base.W_val.multiply(c);
    scaled.W_out = base.W_out.divide(c);
    Tensor x = Tensors.vector(-0.8, 1.4);
    Tolerance.CHOP.requireClose(scaled.forward(x), base.forward(x));
  }
}
