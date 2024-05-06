const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const llama_cpp = try build_llama_cpp(.{ .b = b, .target = target, .optimize = optimize });

    const exe = b.addExecutable(.{
        .name = "llama.cpp-zig",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.addIncludePath(.{ .path = "llama.cpp" });
    exe.linkLibrary(llama_cpp);
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}

fn build_llama_cpp(params: struct { b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode }) !*std.Build.Step.Compile {
    const b = params.b;
    const target = params.target;
    const optimize = params.optimize;

    const commit_hash = try std.ChildProcess.run(.{ .allocator = b.allocator, .argv = &.{ "git", "rev-parse", "HEAD" }, .cwd = b.pathFromRoot("llama.cpp") });
    const zig_version = builtin.zig_version_string;
    try b.build_root.handle.writeFile2(.{ .sub_path = "llama.cpp/common/build-info.cpp", .data = b.fmt(
        \\int LLAMA_BUILD_NUMBER = {};
        \\char const *LLAMA_COMMIT = "{s}";
        \\char const *LLAMA_COMPILER = "Zig {s}";
        \\char const *LLAMA_BUILD_TARGET = "{s}";
        \\
    , .{ 0, commit_hash.stdout[0 .. commit_hash.stdout.len - 1], zig_version, try target.result.zigTriple(b.allocator) }) });

    var objs = std.ArrayList(*std.Build.Step.Compile).init(b.allocator);

    try objs.append(build_obj(.{ .b = b, .name = "ggml", .target = target, .optimize = optimize, .source_file = "llama.cpp/ggml.c" }));
    try objs.append(build_obj(.{ .b = b, .name = "sgemm", .target = target, .optimize = optimize, .source_file = "llama.cpp/sgemm.cpp" }));
    try objs.append(build_obj(.{ .b = b, .name = "ggml_alloc", .target = target, .optimize = optimize, .source_file = "llama.cpp/ggml-alloc.c" }));
    try objs.append(build_obj(.{ .b = b, .name = "ggml_backend", .target = target, .optimize = optimize, .source_file = "llama.cpp/ggml-backend.c" }));
    try objs.append(build_obj(.{ .b = b, .name = "ggml_quants", .target = target, .optimize = optimize, .source_file = "llama.cpp/ggml-quants.c" }));
    try objs.append(build_obj(.{ .b = b, .name = "llama", .target = target, .optimize = optimize, .source_file = "llama.cpp/llama.cpp" }));
    try objs.append(build_obj(.{ .b = b, .name = "unicode", .target = target, .optimize = optimize, .source_file = "llama.cpp/unicode.cpp" }));
    try objs.append(build_obj(.{ .b = b, .name = "unicode_data", .target = target, .optimize = optimize, .source_file = "llama.cpp/unicode-data.cpp" }));
    try objs.append(build_obj(.{ .b = b, .name = "common", .target = target, .optimize = optimize, .source_file = "llama.cpp/common/common.cpp" }));
    try objs.append(build_obj(.{ .b = b, .name = "console", .target = target, .optimize = optimize, .source_file = "llama.cpp/common/console.cpp" }));
    try objs.append(build_obj(.{ .b = b, .name = "sampling", .target = target, .optimize = optimize, .source_file = "llama.cpp/common/sampling.cpp" }));
    try objs.append(build_obj(.{ .b = b, .name = "grammar_parser", .target = target, .optimize = optimize, .source_file = "llama.cpp/common/grammar-parser.cpp" }));
    try objs.append(build_obj(.{ .b = b, .name = "json_schema_to_grammar", .target = target, .optimize = optimize, .source_file = "llama.cpp/common/json-schema-to-grammar.cpp" }));

    const llama_cpp = b.addStaticLibrary(.{ .name = "llama.cpp", .target = target, .optimize = optimize });
    for (objs.items) |obj| {
        llama_cpp.addObject(obj);
    }

    return llama_cpp;
}

fn build_obj(params: struct {
    b: *std.Build,
    name: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    source_file: []const u8,
}) *std.Build.Step.Compile {
    const obj = params.b.addObject(.{
        .target = params.target,
        .optimize = params.optimize,
        .name = "obj",
    });
    obj.addIncludePath(.{ .path = "llama.cpp" });
    obj.addIncludePath(.{ .path = "llama.cpp/common" });
    obj.addCSourceFile(.{ .file = .{ .path = params.source_file } });
    obj.linkLibC();
    obj.linkLibCpp();
    return obj;
}
