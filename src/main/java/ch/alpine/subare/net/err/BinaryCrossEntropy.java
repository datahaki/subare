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
import ch.alpine.tensor.sca.exp.LogisticSigmoid;

public enum BinaryCrossEntropy implements LossFunction, Serializable {
  INSTANCE;

  /** @param logits raw logit (Scalar) or 1D vector of raw logits for multi-label
   * @param target scalar target (0 or 1, or soft label in [0, 1]) or 1D vector matching _p
   * @return loss and gradient with respect to logits _p */
  @Override
  public NetLoss compute(Tensor logits, Tensor target) {
    return _compute(logits, target);
  }

  /** @param logits batch tensor of raw logits (vector of scalars or matrix for multi-label)
   * @param target vector of targets matching batch size
   * @return mean loss and batch-scaled gradient tensor with respect to logits _p */
  @Override
  public NetLoss computeBatch(Tensor logits, Tensor target) {
    int batch_size = Integers.requireEquals(logits.length(), target.length());
    Int myI = new Int();
    List<NetLoss> list = logits.stream() //
        .map(pred -> _compute(pred, target.Get(myI.getAndIncrement()))) //
        .toList();
    Scalar loss = (Scalar) Mean.of(list.stream().map(NetLoss::loss));
    Scalar factor = Rational.of(1, batch_size);
    Tensor gradOutput = Tensor.of(list.stream() //
        .map(NetLoss::gradOutput) //
        .map(grad -> grad.multiply(factor)), batch_size);
    return new NetLoss(loss, gradOutput);
  }

  private static NetLoss _compute(Tensor logitTensor, Tensor targetTensor) {
    if (logitTensor instanceof Scalar z && targetTensor instanceof Scalar y) {
      return new NetLoss(bceFromLogit(z, y), LogisticSigmoid.FUNCTION.apply(z).subtract(y));
    }
    int length = Integers.requireEquals(logitTensor.length(), targetTensor.length());
    Scalar[] losses = new Scalar[length];
    Scalar[] grads = new Scalar[length];
    for (int i = 0; i < length; i++) {
      Scalar z = logitTensor.Get(i);
      Scalar y = targetTensor.Get(i);
      losses[i] = bceFromLogit(z, y);
      grads[i] = LogisticSigmoid.FUNCTION.apply(z).subtract(y);
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
    return "BinaryCrossEntropy";
  }
}
