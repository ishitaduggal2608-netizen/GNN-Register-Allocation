package generator;

import allocator.ChaitinBriggsAllocator;
import cfg.ControlFlowGraph;
import exporter.GraphJsonExporter;
import interference.InterferenceGraphBuildResult;
import interference.InterferenceGraphBuilder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;

/** Generates graph-disjoint train/validation/test JSON datasets from synthetic TAC CFGs. */
public final class DatasetGenerator {
    private static final int[] REGISTER_BUDGETS = {4, 8, 16};

    public void generate(Path outputDirectory, int graphCount, long seed) throws IOException {
        generateRange(outputDirectory, graphCount, 0, graphCount, seed);
    }

    /** Generates a deterministic index range, allowing interrupted runs to resume safely. */
    public void generateRange(Path outputDirectory, int graphCount, int startInclusive,
                              int endExclusive, long seed) throws IOException {
        if (graphCount < 1) throw new IllegalArgumentException("graphCount must be positive");
        if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > graphCount) {
            throw new IllegalArgumentException("Invalid generation range");
        }
        InterferenceGraphBuilder graphBuilder = new InterferenceGraphBuilder();
        ChaitinBriggsAllocator allocator = new ChaitinBriggsAllocator();
        GraphJsonExporter exporter = new GraphJsonExporter();

        for (int graphIndex = startInclusive; graphIndex < endExclusive; graphIndex++) {
            Random random = new Random(seed + 0x9E3779B97F4A7C15L * graphIndex);
            SyntheticCfgGenerator cfgGenerator = new SyntheticCfgGenerator(random);
            // Split is assigned before CFG generation, feature extraction, or allocation.
            String split = splitFor(graphIndex, graphCount);
            int variableCount = 10 + random.nextInt(191); // 10 through 200 nodes before dead-code effects.
            int registerBudget = REGISTER_BUDGETS[graphIndex % REGISTER_BUDGETS.length];
            ControlFlowGraph cfg = cfgGenerator.generate(variableCount);
            InterferenceGraphBuildResult result = graphBuilder.build(cfg);
            var allocation = allocator.allocate(result.getGraph(), registerBudget);
            String graphId = String.format("func_%03d", graphIndex);
            exporter.write(outputDirectory.resolve(split).resolve(graphId + ".json"), graphId,
                    registerBudget, result.getGraph(), allocation);
        }
    }

    public String splitFor(int graphIndex, int graphCount) {
        if (graphIndex < 0 || graphIndex >= graphCount) throw new IllegalArgumentException("Invalid graph index");
        if (graphIndex < graphCount * 70 / 100) return "train";
        if (graphIndex < graphCount * 85 / 100) return "val";
        return "test";
    }

    public static void main(String[] args) throws IOException {
        Path output = args.length > 0 ? Path.of(args[0]) : Path.of("data", "generated");
        int graphCount = args.length > 1 ? Integer.parseInt(args[1]) : 500;
        long seed = args.length > 2 ? Long.parseLong(args[2]) : 42L;
        int startIndex = args.length > 3 ? Integer.parseInt(args[3]) : 0;
        new DatasetGenerator().generateRange(output, graphCount, startIndex, graphCount, seed);
    }
}
