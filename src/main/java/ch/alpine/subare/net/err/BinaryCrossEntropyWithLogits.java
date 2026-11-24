// code by gemini
package ch.alpine.subare.net.err;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Stream;

import ch.alpine.subare.net.LossFunction;
import ch.alpine.subare.net.NetLoss;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.Int;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.red.Max;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.Abs;
import ch.alpine.tensor.sca.exp.Exp;
import ch.alpine.tensor.sca.exp.Log;

public abstract class BinaryCrossEntropyWithLogits implements LossFunction, Serializable {
  /** @param _p sigmoid(logits()) for a single sample (scalar or 1D vector)
   * @param target scalar target (0 or 1, or soft label in [0, 1]) or 1D vector for multi-label
   * @return loss from logits() and gradient using _p minus target */
  @Override
  public NetLoss compute(Tensor _p, Tensor target) {
    return _compute(logits(), _p, target);
  }

  /** @param _p batch tensor of sigmoid(logits()) matching batch size
   * @param target vector of targets matching batch size
   * @return mean loss and batch-scaled gradient tensor with respect to logits() */
  @Override
  public NetLoss computeBatch(Tensor _p, Tensor target) {
    Tensor batchLogits = logits();
    int batchSize = Integers.requireEquals(batchLogits.length(), target.length());
    Integers.requireEquals(_p.length(), batchSize);
    Int index = new Int();
    List<NetLoss> results = batchLogits.stream() //
        .map(logit -> {
          int i = index.getAndIncrement();
          return _compute(logit, _p.Get(i), target.Get(i));
        }) //
        .toList();
    Scalar loss = (Scalar) Mean.of(results.stream().map(NetLoss::loss));
    Scalar factor = Rational.of(1, batchSize);
    Tensor gradOutput = Tensor.of(results.stream() //
        .map(NetLoss::gradOutput) //
        .map(grad -> grad.multiply(factor)), batchSize);
    return new NetLoss(loss, gradOutput);
  }

  /** @return raw unnormalized logits corresponding to _p = sigmoid(logits()) */
  protected abstract Tensor logits();

  private static NetLoss _compute(Tensor logitTensor, Tensor probTensor, Tensor targetTensor) {
    if (logitTensor instanceof Scalar z && probTensor instanceof Scalar p && targetTensor instanceof Scalar y) {
      return new NetLoss(bceFromLogit(z, y), p.subtract(y));
    }
    int length = Integers.requireEquals(logitTensor.length(), targetTensor.length());
    Integers.requireEquals(probTensor.length(), length);
    Scalar[] losses = new Scalar[length];
    Scalar[] grads = new Scalar[length];
    for (int i = 0; i < length; i++) {
      Scalar z = logitTensor.Get(i);
      Scalar p = probTensor.Get(i);
      Scalar y = targetTensor.Get(i);
      losses[i] = bceFromLogit(z, y);
      grads[i] = p.subtract(y);
    }
    return new NetLoss((Scalar) Mean.of(Stream.of(losses)), Tensor.of(Stream.of(grads)));
  }

  private static Scalar bceFromLogit(Scalar z, Scalar y) {
    Scalar maxVal = Max.function(RealScalar.ZERO).apply(z);
    Scalar zTimesY = z.multiply(y);
    Scalar negAbsZ = Exp.FUNCTION.apply(Abs.FUNCTION.apply(z).negate());
    Scalar logTerm = Log.FUNCTION.apply(RealScalar.ONE.add(negAbsZ));
    return maxVal.subtract(zTimesY).add(logTerm);
  }

  @Override
  public String toString() {
    return "BinaryCrossEntropyWithLogits";
  }
}
