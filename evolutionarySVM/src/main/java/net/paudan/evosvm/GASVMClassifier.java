package net.paudan.evosvm;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.jgap.Chromosome;
import org.jgap.Configuration;
import org.jgap.FitnessFunction;
import org.jgap.Gene;
import org.jgap.Genotype;
import org.jgap.IChromosome;
import org.jgap.impl.CrossoverOperator;
import org.jgap.impl.DefaultConfiguration;
import org.jgap.impl.DoubleGene;
import org.jgap.impl.IntegerGene;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;


class MaximizeGAFitnessFunction extends FitnessFunction {

    private final Instances traindata;
    private boolean showDebugInformation;

    public MaximizeGAFitnessFunction(Instances traindata, boolean showDebugInformation) {
        this.traindata = traindata;
        this.showDebugInformation = showDebugInformation;
    }

    @Override
    protected double evaluate(IChromosome a_subject) {
        double sum = 0;
        try {
            weka.classifiers.functions.LibLINEAR class1 = new weka.classifiers.functions.LibLINEAR();
            class1.setSVMType(new SelectedTag(getSVMType(a_subject), weka.classifiers.functions.LibLINEAR.TAGS_SVMTYPE));
            class1.setCost(getC(a_subject));
            class1.setBias(getBias(a_subject));
            class1.setEps(1e-6);
            //class1.buildClassifier(traindata);
            Evaluation eval = new Evaluation(traindata);
            eval.crossValidateModel(class1, traindata, 2, new Random());
            for (int i = 0; i < traindata.numClasses(); i++)
                sum += eval.recall(i);

            a_subject.setFitnessValue(sum);
            if (showDebugInformation)
                System.out.println("Fitness value: "+sum);
        } catch (Exception ex) {
            Logger.getLogger(MaximizeFitnessFunction.class.getName()).log(Level.SEVERE, null, ex);
        }
        return sum;
    }

    public static Integer getSVMType(IChromosome a_potentialSolution) {
        return (Integer) a_potentialSolution.getGene(0).getAllele();
    }

    public static Double getC(IChromosome a_potentialSolution) {
        return (Double) a_potentialSolution.getGene(1).getAllele();
    }

    public static Double getBias(IChromosome a_potentialSolution) {
        return (Double) a_potentialSolution.getGene(2).getAllele();
    }
}
public class GASVMClassifier extends AbstractClassifier implements TechnicalInformationHandler {

    private double select_rate = 0.7;
    private double maxC = 100;
    private double max_bias = 3;
    private double eps = 1e-6;
    private int population_size = 50;
    private int iterations = 5;
    private weka.classifiers.functions.LibLINEAR class1;
    private boolean showDebugInformation = true;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        Configuration conf = new DefaultConfiguration();
        Configuration.reset();
        conf.setFitnessFunction(new MaximizeGAFitnessFunction(data, showDebugInformation));
        //conf.addGeneticOperator(new MutationOperator(conf, 2/3));
        conf.addGeneticOperator(new CrossoverOperator(conf));
        conf.setSelectFromPrevGen(select_rate);
        Gene[] sampleGenes = new Gene[3];
        sampleGenes[0] = new IntegerGene(conf, 0, 3);  // LibLINEAR model
        sampleGenes[1] = new DoubleGene(conf, 0, maxC); // C parameter
        sampleGenes[2] = new DoubleGene(conf, 0, max_bias);   // Bias
        Chromosome sampleChromosome = new Chromosome(conf, sampleGenes);
        conf.setSampleChromosome(sampleChromosome);
        conf.setPopulationSize(population_size);
        Genotype population = Genotype.randomInitialGenotype(conf);

        for (int i = 0; i < iterations; i++) {
            if (showDebugInformation)
                System.out.println("Population No: " + (i+1));
            population.applyGeneticOperators();
            population.evolve();
        }

        IChromosome bestSolutionSoFar = population.getFittestChromosome();

        int bestSVM = MaximizeGAFitnessFunction.getSVMType(bestSolutionSoFar);
        double bestC = MaximizeGAFitnessFunction.getC(bestSolutionSoFar);
        double bestBias = MaximizeGAFitnessFunction.getBias(bestSolutionSoFar);

        if (showDebugInformation) {
            System.out.println("SVM Type: " + bestSVM);
            System.out.println("C parameter: " + bestC);
            System.out.println("Bias: " + bestBias);
            System.out.println("Fitness value: " + bestSolutionSoFar.getFitnessValue());
        }

        class1 = new weka.classifiers.functions.LibLINEAR();
        class1.setSVMType(new SelectedTag(bestSVM, weka.classifiers.functions.LibLINEAR.TAGS_SVMTYPE));
        class1.setCost(bestC);
        class1.setBias(bestBias);
        class1.setEps(eps);
        class1.buildClassifier(data);
    }

    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public double getSelectRate() {
        return select_rate;
    }

    public void setSelectRate(double select_rate) {
        this.select_rate = select_rate;
    }

    public double getMaxC() {
        return maxC;
    }

    public void setMaxC(double maxC) {
        this.maxC = maxC;
    }

    public double getMaxBias() {
        return max_bias;
    }

    public void setMaxBias(double max_bias) {
        this.max_bias = max_bias;
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

    public int getPopulationSize() {
        return population_size;
    }

    public void setPopulationSize(int population_size) {
        this.population_size = population_size;
    }

    public int getIterations() {
        return iterations;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    public boolean getShowDebugInformation() {
        return showDebugInformation;
    }

    public void setShowDebugInformation(boolean showDebugInformation) {
        this.showDebugInformation = showDebugInformation;
    }
}