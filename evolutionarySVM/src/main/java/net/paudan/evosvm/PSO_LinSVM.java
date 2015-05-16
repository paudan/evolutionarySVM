package net.paudan.evosvm;

import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.Arrays;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

public class PSO_LinSVM extends AbstractClassifier implements TechnicalInformationHandler {

    private int noParticles = 20;
    private double upperC = 100;
    private double lowerC = 1;
    private double upperBias = 10;
    private double lowerBias = -5;
    private int terminateIterations;
    private int maxIterations = -1;
    private double[] sigma = new double[3];
    private int minClass = 0;
    private int maxClass = 7;
    private static double weight = 0.8;
    private int numFolds = 5;
    private double c1, c2;
    private LibLINEAR classifier;
    //PSO-LinSVM performance vector
    private ArrayList<Double[]> perfVector = new ArrayList<Double[]>();
    // Best obtained fitness
    private double bestFitness = 0;
    // Number of iterations
    private int numIterations = 0;
    private double[] g_best = new double[noParticles];
    private double eps = 1e-6;

    /**
     * @return the performance vector as ArrayList of Double[4] values for each PSO iteration needed
     * * first value is the classifier encoded as inner LibLINEAR classifier
     * * second value is the C value
     * * third value is bias value
     * * fourth value is the global fitness value at the iteration
     */
    public ArrayList<Double[]> getPerformanceVector() {
        return perfVector;
    }

    /**
     * @return the best fitness value
     */
    public double getBestFitness() {
        return bestFitness;
    }

    /**
     * @return the number of iterations needed for classifier optimization
     */
    public int getNumIterations() {
        return numIterations;
    }

    /**
     * @return set of best parameters
     */
    public double[] getBestParameters() {
        return g_best;
    }

    public double getC1() {
        return c1;
    }

    public void setC1(double c1) {
        this.c1 = c1;
    }

    public double getC2() {
        return c2;
    }

    public void setC2(double c2) {
        this.c2 = c2;
    }

    public int getNumFolds() {
        return numFolds;
    }

    public void setNumFolds(int numFolds) throws NegativeValueException {
        if (numFolds <= 0)
            throw new NegativeValueException("Value of number of folds must be a positive integer");
        this.numFolds = numFolds;
    }

    public double getLowerBias() {
        return lowerBias;
    }

    public void setLowerBias(double lowerBias) {
        this.lowerBias = lowerBias;
    }

    public double getLowerC() {
        return lowerC;
    }

    public void setLowerC(double lowerC) throws NegativeValueException {
        if (lowerC <= 0)
            throw new NegativeValueException("Value of C parameter must be more than 0");
        this.lowerC = lowerC;
    }

    public int getMaxIterations() {
        return maxIterations;
    }

    public void setMaxIterations(int maxIterations) throws NegativeValueException {
        if (maxIterations <= 0)
            throw new NegativeValueException("The number of iterations must be a positive integer");
        this.maxIterations = maxIterations;
    }

    public int getNoParticles() {
        return noParticles;
    }

    public void setNoParticles(int noParticles) throws NegativeValueException {
        if (noParticles <= 0)
            throw new NegativeValueException("The number of particles must be a positive integer");
        this.noParticles = noParticles;
    }

    public int getTerminateIterations() {
        return terminateIterations;
    }

    public void setTerminateIterations(int terminateIterations) throws NegativeValueException {
        if (terminateIterations <= 0)
            throw new NegativeValueException("The number of iterations must be a positive integer");
        this.terminateIterations = terminateIterations;
    }

    public double getUpperBias() {
        return upperBias;
    }

    public void setUpperBias(double upperBias) {
        this.upperBias = upperBias;
    }

    public double getUpperC() {
        return upperC;
    }

    public void setUpperC(double upperC) {
        this.upperC = upperC;
    }

    public void buildClassifier(Instances data) throws Exception {
        int dimensions = 3;
        double global_fitness = 0.0;
        int term_iterations = 1;
        int count = 0;

        double lowerb[] = new double[3];
        lowerb[0] = minClass;
        lowerb[1] = lowerC;
        lowerb[2] = lowerBias;
        double upperb[] = new double[3];
        upperb[0] = maxClass;
        upperb[1] = upperC;
        upperb[2] = upperBias;
        for (int i = 0; i < 3; i++)
            sigma[i] = upperb[i] / (Math.abs(lowerb[i]) + Math.abs(upperb[i])) * weight;
        
        int[] classinst = data.attributeStats(data.classIndex()).nominalCounts;

        double[][] particle_position = new double[noParticles][dimensions];
        double[][] p_best = new double[noParticles][dimensions];
        double[] p_best_fitness = new double[noParticles];
        double[][] particle_velocity = new double[noParticles][dimensions];
        double[] current_fitness = new double[noParticles];
        double[] p_current = new double[dimensions];

        // Initialize particles and velocity components
        SecureRandom rand = new SecureRandom();
        for (int count_x = 0; count_x < noParticles; count_x++) {
            particle_position[count_x][0] = lowerb[0] + Math.round(rand.nextDouble() * (upperb[0] - lowerb[0]));
            long classval = new Double(particle_position[count_x][0]).longValue();
            if (classval == 4) {
                while (classval == 4)
                    classval = Math.round(lowerb[0] + Math.round(rand.nextDouble() * (upperb[0] - lowerb[0])));
                particle_position[count_x][0] = classval;
            }
            for (int i = 1; i < dimensions; i++)
                particle_position[count_x][i] = lowerb[i] + rand.nextDouble() * (upperb[i] - lowerb[i]);
            for (int i = 0; i < dimensions; i++) {
                p_best[count_x][i] = particle_position[count_x][i];
                particle_velocity[count_x][i] = 0.0;
            }
            // Initialize best fitness array
            p_best_fitness[count_x] = -1000;
        }

        // Main routine
        while (term_iterations < terminateIterations) {
            count++;

            System.out.println("Iteration:" + count);
            /*for (int count_x = 0; count_x < noParticles; count_x++) {
                for (int i = 0; i < dimensions; i++)
                    System.out.print(particle_position[count_x][i] + " ");
                System.out.println();
            }*/
            if (count == maxIterations)
                break;

            // find the fitness of each particle
            for (int count_x = 0; count_x < noParticles; count_x++) {
                LibLINEAR svm = new LibLINEAR();
                svm.setSVMType(new SelectedTag((int) particle_position[count_x][0], LibLINEAR.TAGS_SVMTYPE));
                svm.setCost(particle_position[count_x][1]);
                svm.setBias(particle_position[count_x][2]);
                svm.setBias(eps);
                Evaluation eval = new Evaluation(data);
                eval.crossValidateModel(svm, data, numFolds, new SecureRandom());
                double sum = 0.0;
                for (int i = 0; i < data.numClasses(); i++)
                    sum += eval.recall(i);
               current_fitness[count_x] = sum;
            }
            System.out.println(Arrays.toString(current_fitness));
            // decide on p_best for each particle
            for (int count_x = 0; count_x < noParticles; count_x++)
                if (current_fitness[count_x] > p_best_fitness[count_x]) {
                    p_best_fitness[count_x] = current_fitness[count_x];
                    for (int count_y = 0; count_y < dimensions; count_y++)
                        p_best[count_x][count_y] = particle_position[count_x][count_y];
                }

            // decide on the global best among all the particles
            double g_best_val = current_fitness[0];
            int g_best_index = 0;
            for (int i = 1; i < noParticles; i++)
                if (current_fitness[i] > g_best_val) {
                    g_best_val = current_fitness[i];
                    g_best_index = i;
                }
            // If global_fitness (found so far) is less than global fitness among current particles
            if (global_fitness < g_best_val) {
                // Update global fitness value
                global_fitness = g_best_val;
                // Update g_best which contains the position of the global best
                for (int count_y = 0; count_y < dimensions; count_y++)
                    g_best[count_y] = particle_position[g_best_index][count_y];
                // As better solution was found, termination flag is reset
                term_iterations = 1;
            } else
                // No improvement
                term_iterations += 1;

            // update the position and velocity compponents
            for (int count_x = 0; count_x < noParticles; count_x++) {
                for (int count_y = 0; count_y < dimensions; count_y++)
                    p_current[count_y] = particle_position[count_x][count_y];

                // Try to implement velocity clamping
                for (int j = 0; j < dimensions; j++) {
                    // Calculate maximum allowed velocity
                    double Vmax = sigma[j] * (upperb[j] - lowerb[j]);
                    double velocity = 0.0;
                    if (j == 0) {
                        Vmax = Math.round(Vmax);
                        // Calculate velocity
                        velocity = particle_velocity[count_x][j] +
                                Math.round(c1 * rand.nextDouble() * Math.abs(p_best[count_x][j] - p_current[j])) +
                                Math.round(c2 * rand.nextDouble() * Math.abs(g_best[j] - p_current[j]));
                    } else
                        velocity = particle_velocity[count_x][j] +
                                c1 * rand.nextDouble() * (p_best[count_x][j] - p_current[j]) +
                                c2 * rand.nextDouble() * (g_best[j] - p_current[j]);
                    // particle_velocity(count_x, j) = Vmax * tanh(velocity/Vmax);
                    if (velocity < Vmax)
                        particle_velocity[count_x][j] = velocity;
                    else
                        particle_velocity[count_x][j] = Vmax;
                }

                for (int count_y = 0; count_y < dimensions; count_y++)
                    particle_position[count_x][count_y] = p_current[count_y] + particle_velocity[count_x][count_y];

                // If particle reaches boundaries for first dimension it "bounces"
                if (particle_position[count_x][0] > upperb[0])
                    particle_position[count_x][0] = particle_position[count_x][0] % upperb[0];

                // If particle reaches position 4 in first dimension it jumps to 5
                particle_position[count_x][0] = (particle_position[count_x][0] == 4 ? 5 : particle_position[count_x][0]);

                // If value of dimension 2 is less than zero it is replaced with lowest feasible value
                if (particle_position[count_x][1] < lowerb[1])
                    particle_position[count_x][1] = lowerb[1];
            }

            // Add main statistics to performance vector
            Double[] perf = new Double[4];
            for (int i = 0; i < dimensions; i++)
                perf[i] = g_best[i];
            perf[dimensions] = global_fitness;
            perfVector.add(perf);
        }

        classifier = new LibLINEAR();
        classifier.setSVMType(new SelectedTag(new Double(g_best[0]).intValue(), LibLINEAR.TAGS_SVMTYPE));
        classifier.setCost(g_best[1]);
        classifier.setBias(g_best[2]);
        classifier.setEps(eps);
        classifier.buildClassifier(data);

        // Perform cleaning
        particle_position = null;
        p_best = null;
        p_best_fitness = null;
        particle_velocity = null;
        current_fitness = null;
        p_current = null;
        System.gc();
    }

    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return classifier.distributionForInstance(instance);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return classifier.classifyInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        return classifier.getCapabilities();
    }

    @Override
    public String[] getOptions() {
        return classifier.getOptions();
    }
}
