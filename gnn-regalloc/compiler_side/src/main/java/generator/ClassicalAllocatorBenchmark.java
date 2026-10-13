package generator;

import allocator.AllocationResult;
import allocator.ChaitinBriggsAllocator;
import cfg.ControlFlowGraph;
import interference.InterferenceGraphBuildResult;
import interference.InterferenceGraphBuilder;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/** Measures only the Java Chaitin-Briggs allocation stage for synthetic graph IDs. */
public final class ClassicalAllocatorBenchmark {
    private static final int[] REGISTER_BUDGETS = {4, 8, 16};

    private ClassicalAllocatorBenchmark() { }

    public static void main(String[] args) throws IOException {
        Path output = args.length > 0 ? Path.of(args[0]) : Path.of("results", "classical_benchmark.csv");
        int graphCount = args.length > 1 ? Integer.parseInt(args[1]) : 500;
        long seed = args.length > 2 ? Long.parseLong(args[2]) : 42L;
        int start = args.length > 3 ? Integer.parseInt(args[3]) : graphCount * 85 / 100;
        int end = args.length > 4 ? Integer.parseInt(args[4]) : graphCount;
        if (start < 0 || end < start || end > graphCount) throw new IllegalArgumentException("Invalid range");

        InterferenceGraphBuilder builder = new InterferenceGraphBuilder();
        ChaitinBriggsAllocator allocator = new ChaitinBriggsAllocator();
        StringBuilder csv = new StringBuilder("graph_id,classical_spills,classical_allocator_seconds\n");
        for (int graphIndex = start; graphIndex < end; graphIndex++) {
            Random random = new Random(seed + 0x9E3779B97F4A7C15L * graphIndex);
            int variableCount = 10 + random.nextInt(191);
            ControlFlowGraph cfg = new SyntheticCfgGenerator(random).generate(variableCount);
            InterferenceGraphBuildResult build = builder.build(cfg);
            long started = System.nanoTime();
            AllocationResult allocation = allocator.allocate(build.getGraph(),
                    REGISTER_BUDGETS[graphIndex % REGISTER_BUDGETS.length]);
            double seconds = (System.nanoTime() - started) / 1_000_000_000.0;
            int spills = (int) build.getGraph().getNodes().stream().filter(node -> node.isLabelSpill()).count();
            csv.append(String.format("func_%03d,%d,%.9f%n", graphIndex, spills, seconds));
        }
        Path parent = output.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
        Files.writeString(output, csv.toString(), StandardCharsets.UTF_8);
    }
}
