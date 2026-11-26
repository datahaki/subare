// code by jph
package ch.alpine.subare.net.mod;

import java.util.List;
import java.util.stream.Stream;

import ch.alpine.subare.net.lay.ArgMaxLayer;
import ch.alpine.subare.net.lay.NetChain;
import ch.alpine.subare.net.lay.NetLayer;
import ch.alpine.subare.net.lay.NetLayers;
import ch.alpine.subare.net.lay.SigmoidLayer;
import ch.alpine.subare.net.lay.SoftmaxLayer;
import ch.alpine.subare.net.lay.UnitStepLayer;

public enum NetModels {
  ;
  /** the output of netLayer is a vector of length n
   * where n corresponds to the number of labels for classification
   * 
   * @param netLayers
   * @return layer that outputs vector of length 1 with integer value as entry */
  public static NetModel softmax_argMax(NetLayer... netLayers) {
    ArgMaxLayer argMaxLayer = new ArgMaxLayer();
    return new NetModel( //
        NetChain.join(Stream.concat( //
            Stream.of(netLayers), //
            Stream.of( //
                SoftmaxLayer.INSTANCE, //
                argMaxLayer))), //
        argMaxLayer.crossEntropyLoss());
  }

  public static NetModel logits(NetLayer... netLayers) {
    UnitStepLayer unitStepLayer = new UnitStepLayer();
    return new NetModel( //
        NetChain.join(Stream.concat( //
            Stream.of(netLayers), //
            Stream.of(unitStepLayer))), //
        unitStepLayer.lossFunction());
  }

  /** intended for binary classification tasks
   * even though network layer does not round output to integer values (!)
   * 
   * @param netLayers with final layer that returns a vector of length 1
   * @return */
  public static NetModel sigmoid(NetLayer... netLayers) {
    SigmoidLayer sigmoidLayer = new SigmoidLayer();
    return new NetModel( //
        NetChain.join(Stream.concat( //
            Stream.of(netLayers), //
            Stream.of(sigmoidLayer))), //
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
