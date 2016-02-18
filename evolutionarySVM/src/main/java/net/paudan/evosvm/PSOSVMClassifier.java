package net.paudan.evosvm;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import net.sourceforge.jswarm_pso.FitnessFunction;
import net.sourceforge.jswarm_pso.Particle;
import net.sourceforge.jswarm_pso.Swarm;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibLINEAR;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

class MaximizeFitnessFunction extends FitnessFunction {

    private Instances trainData;
    private boolean showDebugInformation;

    @Override
    public double evaluate(double[] position) {
        double sum = 0;
        if (position[1] > 0) {
            try {
                LibLINEAR class1 = new LibLINEAR();
                class1.setSVMType(new SelectedTag((int) position[0], LibLINEAR.TAGS_SVMTYPE));
                class1.setCost(position[1]);
                class1.setBias(position[2]);
                class1.setEps(1e-6);
                Evaluation eval = new Evaluation(trainData);
                eval.crossValidateModel(class1, trainData, 2, new Random());
                for (int i = 0; i < getTrainData().numClasses(); i++)
                    sum += eval.recall(i);
                System.out.println(sum);
            } catch (Exception ex) {
                Logger.getLogger(MaximizeFitnessFunction.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        return sum;
    }

    public Instances getTrainData() {
        return trainData;
    }

    public void setTrainData(Instances trainData) {
        this.trainData = trainData;
    }

    public boolean showDebugInformation() {
        return showDebugInformation;
    }

    public void setShowDebugInformation(boolean showDebugInformation) {
        this.showDebugInformation = showDebugInformation;
    }
}

public class PSOSVMClassifier extends AbstractClassifier implements TechnicalInformationHandler {

    private double maxC = 100;
    private double maxBias = 5;
    private double minVelocityC = 0.2;
    private double minVelocityBias = 0.05;
    private double maxVelocityC = 0.2;
    private double maxVelocityBias = 0.05;
    private int numParticles = Swarm.DEFAULT_NUMBER_OF_PARTICLES;
    private double inertia = 1;
    private int numIterations = 5;
    private LibLINEAR class1;
    private double eps = 1e-6;
    private boolean showDebugInformation = true;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        MaximizeFitnessFunction fitnessFunction = new MaximizeFitnessFunction();
        fitnessFunction.setTrainData(data);
        fitnessFunction.setShowDebugInformation(showDebugInformation);
        fitnessFunction.setMaximize(true);
        Swarm swarm = new Swarm(numParticles, new SVMParticle(), fitnessFunction);
        swarm.setMinPosition(new double[]{0, 0, 0});
        swarm.setMaxPosition(new double[]{3, maxC, maxBias});
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
        int bestSVM = (int) bestSolutionSoFar.getPosition()[0];
        double bestC = bestSolutionSoFar.getPosition()[1];
        double bestBias = bestSolutionSoFar.getPosition()[2];

        if (showDebugInformation) {
            System.out.println("SVM Type: " + bestSVM);
            System.out.println("C parameter: " + bestC);
            System.out.println("Bias: " + bestBias);
            System.out.println("Fitness value: " + bestSolutionSoFar.getBestFitness());
        }

        class1 = new LibLINEAR();
        class1.setSVMType(new SelectedTag(bestSVM, LibLINEAR.TAGS_SVMTYPE));
        class1.setCost(bestC);
        class1.setBias(bestBias);
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

    public double getMaxBias() {
        return maxBias;
    }

    public void setMaxBias(double maxBias) {
        this.maxBias = maxBias;
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
}
