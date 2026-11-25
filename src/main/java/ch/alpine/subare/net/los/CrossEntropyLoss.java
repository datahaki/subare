// code by jph
package ch.alpine.subare.net.los;

import java.io.Serializable;
import java.util.List;

import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.ScalarUnaryOperator;
import ch.alpine.tensor.ext.Int;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.red.Max;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.Sign;
import ch.alpine.tensor.sca.exp.Log;

public enum CrossEntropyLoss implements LossFunction, Serializable {
  SCALAR {
    /** @param probabilities
     * @param target is a scalar of integer value between 0, 1, 2, and "number of classes"
     * @return */
    @Override
    public NetLoss compute(Tensor probabilities, Tensor target) {
      int k = Scalars.intValueExact((Scalar) target);
      Scalar loss = safeLog(probabilities.Get(k)).negate();
      Tensor gradOutput = probabilities.copy();
      gradOutput.set(RealScalar.ONE.negate()::add, k);
      return new NetLoss(loss, gradOutput);
    }

    /** @param _p is discarded, probabilities() will be queried instead
     * @param target vector of integers
     * @return */
    @Override
    public NetLoss computeBatch(Tensor prediction, Tensor target) {
      int batch_size = Integers.requireEquals(prediction.length(), target.length());
      Int myI = new Int();
      List<NetLoss> list = prediction.stream() //
          .map(pred -> compute(pred, target.Get(myI.getAndIncrement()))) //
          .toList();
      Scalar loss = (Scalar) Mean.of(list.stream().map(NetLoss::loss));
      Scalar factor = Rational.of(1, batch_size);
      Tensor gradOutput = Tensor.of(list.stream() //
          .map(NetLoss::gradOutput) //
          .map(grad -> grad.multiply(factor)), batch_size);
      return new NetLoss(loss, gradOutput);
    }
  };

  private static final ScalarUnaryOperator MAX = Max.function(RealScalar.of(1e-15));

  private static Scalar safeLog(Scalar value) {
    return Log.FUNCTION.apply(MAX.apply(Sign.requirePositiveOrZero(value)));
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("CrossEntropyLoss", name());
  }
}
