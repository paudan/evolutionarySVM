package net.paudan.evosvm;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import net.sourceforge.jswarm_pso.FitnessFunction;
import net.sourceforge.jswarm_pso.Particle;
import net.sourceforge.jswarm_pso.Swarm;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

public class PSOSVMRBFClassifier extends AbstractClassifier implements TechnicalInformationHandler {

    private double minC = 1;
    private double maxC = 100;
    private double minGamma = -10;
    private double maxGamma = 20;
    private double minVelocityC = 0.2;
    private double minVelocityBias = 0.05;
    private double maxVelocityC = 0.2;
    private double maxVelocityBias = 0.05;
    private int numParticles = Swarm.DEFAULT_NUMBER_OF_PARTICLES;
    private double inertia = 1;
    private int numIterations = 5;
    private LibSVM class1;
    private double eps = 1e-6;
    private boolean showDebugInformation = true;

    private class MaximizeFitnessFunction extends FitnessFunction {

        private Instances trainData;
        private boolean showDebugInformation;

        @Override
        public double evaluate(double[] position) {
            double sum = 0;
            try {
                LibSVM class1 = new LibSVM();
                class1.setSVMType(new SelectedTag(LibSVM.SVMTYPE_C_SVC, LibSVM.TAGS_KERNELTYPE));
                class1.setCost(position[0]);
                class1.setGamma(position[1]);
                class1.setEps(getEps());;
                class1.setEps(1e-6);
                //class1.buildClassifier(getTrainData());
                Evaluation eval = new Evaluation(trainData);
                eval.crossValidateModel(class1, trainData, 3, new Random());
                for (int i = 0; i < getTrainData().numClasses(); i++)
                    sum += eval.recall(i);
                System.out.println(sum);
            } catch (Exception ex) {
                Logger.getLogger(MaximizeFitnessFunction.class.getName()).log(Level.SEVERE, null, ex);
            }
            return sum;
        }

        public boolean showDebugInformation() {
            return showDebugInformation;
        }

        public void setShowDebugInformation(boolean showDebugInformation) {
            this.showDebugInformation = showDebugInformation;
        }

        public Instances getTrainData() {
            return trainData;
        }

        public void setTrainData(Instances trainData) {
            this.trainData = trainData;
        }
    }

    public void buildClassifier(Instances data) throws Exception {
        MaximizeFitnessFunction fitnessFunction = new MaximizeFitnessFunction();
        fitnessFunction.setTrainData(data);
        fitnessFunction.setShowDebugInformation(showDebugInformation);
        fitnessFunction.setMaximize(true);
        Swarm swarm = new Swarm(numParticles, new SVMParticle(), fitnessFunction);
        swarm.setMinPosition(new double[]{minC, minGamma});
        swarm.setMaxPosition(new double[]{maxC, maxGamma});
        swarm.setMinVelocity(new double[]{1, minVelocityC, minVelocityBias});
        swarm.setMaxVelocity(new double[]{1, maxVelocityC, maxVelocityBias});
        swarm.setInertia(inertia);
        for (int i = 0; i < numIterations; i++) {
            if (showDebugInformation)
                System.out.println("Iteration: " + (i + 1));
            swarm.evolve();
        }
        if (showDebugInformation)
            System.out.println(swarm.toStringStats());

        Particle bestSolutionSoFar = swarm.getBestParticle();
        double bestC = bestSolutionSoFar.getPosition()[0];
        double bestGamma = bestSolutionSoFar.getPosition()[1];

        if (showDebugInformation) {
            System.out.println("C parameter: " + bestC);
            System.out.println("Bias: " + bestGamma);
            System.out.println("Fitness value: " + bestSolutionSoFar.getBestFitness());
        }

        class1 = new LibSVM();
        class1.setSVMType(new SelectedTag(LibSVM.SVMTYPE_C_SVC, LibSVM.TAGS_KERNELTYPE));
        class1.setCost(bestC);
        class1.setGamma(maxGamma);
        class1.setEps(getEps());
        class1.buildClassifier(data);
    }

    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public double getMaxC() {
        return maxC;
    }

    public void setMaxC(double maxC) {
        this.maxC = maxC;
    }

    public double getMinVelocityC() {
        return minVelocityC;
    }

    public void setMinVelocityC(double minVelocityC) {
        this.minVelocityC = minVelocityC;
    }

    public double getMinVelocityBias() {
        return minVelocityBias;
    }

    public void setMinVelocityBias(double minVelocityBias) {
        this.minVelocityBias = minVelocityBias;
    }

    public int getNumParticles() {
        return numParticles;
    }

    public void setNumParticles(int numParticles) {
        this.numParticles = numParticles;
    }

    public double getInertia() {
        return inertia;
    }

    public void setInertia(double inertia) {
        this.inertia = inertia;
    }

    public int getNumIterations() {
        return numIterations;
    }

    public void setNumIterations(int numIterations) {
        this.numIterations = numIterations;
    }

    public double getEps() {
        return eps;
    }

    public void setEps(double eps) {
        this.eps = eps;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return class1.distributionForInstance(instance);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return class1.classifyInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        return class1.getCapabilities();
    }

    @Override
    public String[] getOptions() {
        return class1.getOptions();
    }

    public boolean showDebugInformation() {
        return showDebugInformation;
    }

    public void setShowDebugInformation(boolean showDebugInformation) {
        this.showDebugInformation = showDebugInformation;
    }

    public double getMaxVelocityC() {
        return maxVelocityC;
    }

    public void setMaxVelocityC(double maxVelocityC) {
        this.maxVelocityC = maxVelocityC;
    }

    public double getMaxVelocityBias() {
        return maxVelocityBias;
    }

    public void setMaxVelocityBias(double maxVelocityBias) {
        this.maxVelocityBias = maxVelocityBias;
    }

    public double getMinC() {
        return minC;
    }

    public void setMinC(double minC) {
        this.minC = minC;
    }

    public double getMinGamma() {
        return minGamma;
    }

    public void setMinGamma(double minGamma) {
        this.minGamma = minGamma;
    }

    public double getMaxGamma() {
        return maxGamma;
    }

    public void setMaxGamma(double maxGamma) {
        this.maxGamma = maxGamma;
    }
}
