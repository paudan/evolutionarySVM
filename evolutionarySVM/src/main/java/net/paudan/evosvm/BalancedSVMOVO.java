package net.paudan.evosvm;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.LibLINEAR;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.meta.Vote;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

public class BalancedSVMOVO extends AbstractClassifier implements TechnicalInformationHandler {

    private AbstractClassifier classifier, generatedClassifier;
    private int svmType;
    private int svmKernel;
    private static double ridge = 20;

    public BalancedSVMOVO() {
        LibSVM class1 = new LibSVM();
        class1.setSVMType(new SelectedTag(LibSVM.SVMTYPE_C_SVC, LibSVM.TAGS_SVMTYPE));
        class1.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_RBF, LibSVM.TAGS_KERNELTYPE));
        class1.setCacheSize(500);
        class1.setShrinking(false);
        class1.setProbabilityEstimates(true);
        class1.setEps(1e-10);
        classifier = class1;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {

        if (data.numClasses() > 2) {
            Vote vote = new Vote();
            vote.setCombinationRule(new SelectedTag(Vote.AVERAGE_RULE, Vote.TAGS_RULES));
            // Create classifiers for each pair of classifiers
            ArrayList classes = Collections.list(data.classAttribute().enumerateValues());
            ArrayList<String> copyclasses = new ArrayList<String>();
            for (Object o : classes)
                copyclasses.add(new String(o.toString()));
            //Calculate ridge (epsilon)
            int[] counts = data.attributeStats(data.classIndex()).nominalCounts;

            int min = counts[0];
            for (int i = 1; i < counts.length; i++)
                if (counts[i] < min)
                    min = counts[i];
            ArrayList<Double> weights = new ArrayList<Double>();
            for (int i = 0; i < counts.length; i++)
                weights.add(ridge * counts[i] / min);

            ArrayList<String> val1 = new ArrayList<String>();
            ArrayList<String> val2 = new ArrayList<String>();
            ArrayList<Double> balweights = new ArrayList<Double>();
            for (int i = 0; i < copyclasses.size(); i++) {
                for (int j = 1; j < copyclasses.size(); j++) {
                    val1.add(copyclasses.get(0));
                    val2.add(copyclasses.get(j));
                    balweights.add(weights.get(0) + weights.get(j));
                }
                copyclasses.remove(copyclasses.get(0));
                weights.remove(weights.get(0));
            }
            System.out.println(Arrays.toString(balweights.toArray()));
            Classifier[] classifiers = new Classifier[val1.size()];
            for (int i = 0; i < val1.size(); i++) {
                try {
                    Instances binarytrain = copyInstances(data);
                    Attribute classattr = binarytrain.classAttribute();
                    for (int j = binarytrain.numInstances() - 1; j >= 0; j--)
                        if (binarytrain.get(j).stringValue(classattr).compareTo(val1.get(i)) != 0 &&
                                binarytrain.get(j).stringValue(classattr).compareTo(val2.get(i)) != 0)
                            binarytrain.remove(j);
                    Classifier class1 = AbstractClassifier.makeCopy(classifier);
                    configureClassifier(class1, balweights.get(i));
                    class1.buildClassifier(binarytrain);
                    classifiers[i] = class1;
                } catch (Exception ex) {
                    Logger.getLogger(BalancedSVMOVO.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
            vote.setClassifiers(classifiers);
            generatedClassifier = vote;
        } else {
            Classifier class1 = AbstractClassifier.makeCopy(classifier);
            configureClassifier(class1, 100);
            class1.buildClassifier(data);
            generatedClassifier = (AbstractClassifier)class1;
        }
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public void setClassifier(AbstractClassifier classifier) {
        this.classifier = classifier;
    }

    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public int getSvmType() {
        return svmType;
    }

    public void setSvmType(int svmType) throws SVMParameterException {
        this.svmType = svmType;
        if (classifier != null)
            if (classifier instanceof LibSVM)
                if (svmType == LibSVM.SVMTYPE_C_SVC || svmType == LibSVM.SVMTYPE_NU_SVC)
                    ((LibSVM) classifier).setSVMType(new SelectedTag(svmType, LibSVM.TAGS_SVMTYPE));
                else
                    throw new SVMParameterException("LibSVM classifier can be only of LibSVM.SVMTYPE_C_SVC or LibSVM.SVMTYPE_NU_SVC type");
            else if (classifier instanceof LibLINEAR)
                if (svmType == LibLINEAR.SVMTYPE_L1LOSS_SVM_DUAL || svmType == LibLINEAR.SVMTYPE_L2LOSS_SVM ||
                        svmType == LibLINEAR.SVMTYPE_L2LOSS_SVM_DUAL || svmType == LibLINEAR.SVMTYPE_MCSVM_CS)
                    ((LibLINEAR) classifier).setSVMType(new SelectedTag(svmType, LibLINEAR.TAGS_SVMTYPE));
                else
                    throw new SVMParameterException("LibLINEAR SVM classifier can be only of LibLINEAR classifier types");
    }

    public int getSvmKernel() {
        return svmKernel;
    }

    public void setSvmKernel(int svmKernel) throws SVMParameterException {
        this.svmKernel = svmKernel;
        if (classifier != null)
            if (classifier instanceof LibSVM)
                if (svmKernel == LibSVM.KERNELTYPE_LINEAR || svmKernel == LibSVM.KERNELTYPE_POLYNOMIAL ||
                        svmKernel == LibSVM.KERNELTYPE_RBF || svmKernel == LibSVM.KERNELTYPE_SIGMOID)
                    ((LibSVM) classifier).setKernelType(new SelectedTag(svmKernel, LibSVM.TAGS_KERNELTYPE));
                else
                    throw new SVMParameterException("LibSVM kernel can be only of LibSVM defined kernel types");
    }

    private Instances copyInstances(Instances traindata) {
        ObjectOutputStream out = null;
        Instances copyinst = null;
        try {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            out = new ObjectOutputStream(bos);
            out.writeObject(traindata);
            out.flush();
            out.close();
            // Make an input stream from the byte array and read
            // a copy of the object back in.
            ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bos.toByteArray()));
            copyinst = (Instances) in.readObject();
        } catch (ClassNotFoundException ex) {
            Logger.getLogger(BalancedSVMOVO.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(BalancedSVMOVO.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                out.close();
            } catch (IOException ex) {
                Logger.getLogger(BalancedSVMOVO.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        return copyinst;
    }

    private void configureClassifier(Classifier class1, double cost) {
        if (class1 instanceof LibSVM) {
            LibSVM libsvm = (LibSVM) class1;
            libsvm.setCost(cost);
            libsvm.setEps(1e-10);
        } else if (class1 instanceof LibLINEAR) {
            LibLINEAR liblinear = (LibLINEAR) class1;
            liblinear.setCost(cost);
            liblinear.setEps(1e-10);
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return generatedClassifier.distributionForInstance(instance);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return generatedClassifier.classifyInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        return generatedClassifier.getCapabilities();
    }

    @Override
    public String[] getOptions() {
        return generatedClassifier.getOptions();
    }
}
