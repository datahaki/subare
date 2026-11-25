// code by jph
package ch.alpine.subare.net;

import java.util.List;

import ch.alpine.subare.net.lay.ArgMaxLayer;
import ch.alpine.subare.net.lay.ScalarLayer;
import ch.alpine.subare.net.lay.SigmoidLayer;
import ch.alpine.subare.net.lay.SoftmaxLayer;
import ch.alpine.subare.net.lay.UnitStepLayer;

public enum NetModels {
  ;
  /** the output of netLayer is a vector of length n
   * where n corresponds to the number of labels for classification
   * 
   * @param netLayer
   * @return layer that outputs vector of length 1 with integer value as entry */
  public static NetModel softmax_argMax(NetLayer netLayer) {
    ArgMaxLayer argMaxLayer = new ArgMaxLayer();
    return new NetModel( //
        NetChain.join( //
            netLayer, //
            SoftmaxLayer.INSTANCE, //
            argMaxLayer), // maps vector to scalar
        argMaxLayer.crossEntropyLoss());
  }

  public static NetModel logits(NetLayer netLayer) {
    UnitStepLayer unitStepLayer = new UnitStepLayer();
    return new NetModel( //
        NetChain.join( //
            netLayer, //
            ScalarLayer.VectorToScalar, //
            unitStepLayer), //
        unitStepLayer.lossFunction());
  }

  /** intended for binary classification tasks
   * even though network layer does not round output to integer values (!)
   * 
   * @param netLayer with final layer that returns a vector of length 1
   * @return */
  public static NetModel sigmoid(NetLayer netLayer) {
    SigmoidLayer sigmoidLayer = new SigmoidLayer();
    return new NetModel( //
        NetChain.join( //
            netLayer, //
            ScalarLayer.VectorToScalar, //
            sigmoidLayer), //
        sigmoidLayer.lossFunction());
  }

  /** @param netLayer
   * @return */
  public static List<String> toString(NetModel netModel) {
    List<String> list = NetLayers.toString(netModel.netLayer());
    list.add(netModel.lossFunction().toString());
    return list;
  }
}
