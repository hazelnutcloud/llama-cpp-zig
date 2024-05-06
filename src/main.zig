const std = @import("std");
const c = @cImport({
    @cInclude("llama.h");
});

pub fn main() !void {
    const console = std.io.getStdOut().writer();
    c.llama_backend_init();
    const model = c.llama_load_model_from_file("phi-3-mini-128k-instruct.q5_k_m.gguf", c.llama_model_default_params());

    if (model == null) {
        std.debug.print("Failed to load model\n", .{});
        return;
    }

    const input = "My name is ";
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer {
        if (gpa.deinit() == .leak) {
            std.debug.print("Memory leak detected\n", .{});
        }
    }

    var tokens = try std.ArrayList(c.llama_token).initCapacity(allocator, input.len + 2);
    defer tokens.deinit();

    const n_tokens = c.llama_tokenize(model, input, input.len, tokens.items.ptr, @intCast(tokens.items.len), true, false);
    if (n_tokens < 0) {
        try tokens.resize(@intCast(-n_tokens));
        const check = c.llama_tokenize(model, input, input.len, tokens.items.ptr, @intCast(tokens.items.len), true, false);
        std.debug.assert(check == -n_tokens);
    } else {
        try tokens.resize(@intCast(n_tokens));
    }

    const ctx = c.llama_new_context_with_model(model, c.llama_context_default_params());
    var batch = c.llama_batch_init(512, 0, 1);

    for (tokens.items, 0..) |token, idx| {
        batch.token[idx] = token;
        batch.pos[idx] = @intCast(idx);
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = 0;
        batch.logits[idx] = @intFromBool(false);
        batch.n_tokens += 1;
    }

    batch.logits[@intCast(batch.n_tokens - 1)] = @intFromBool(true);

    const decode_res = c.llama_decode(ctx, batch);
    if (decode_res != 0) {
        std.debug.print("Failed to decode: {d}\n", .{decode_res});
        return;
    }

    const n_len = 512;
    var n_cur = batch.n_tokens;

    while (n_cur <= n_len) {
        const n_vocab: usize = @intCast(c.llama_n_vocab(model));
        const logits = c.llama_get_logits_ith(ctx, batch.n_tokens - 1);

        var candidates = try std.ArrayList(c.llama_token_data).initCapacity(allocator, n_vocab);
        defer candidates.deinit();

        var token_id: i32 = 0;
        for (0..n_vocab) |idx| {
            try candidates.append(c.llama_token_data{ .id = token_id, .logit = logits[idx], .p = 0.0 });
            token_id += 1;
        }

        var candidates_p = c.llama_token_data_array{ .data = candidates.items.ptr, .size = candidates.items.len, .sorted = false };

        const new_token_id = c.llama_sample_token_greedy(ctx, &candidates_p);

        if (c.llama_token_is_eog(model, new_token_id)) {
            try console.print("\n", .{});
            break;
        }

        var buf = try std.ArrayList(u8).initCapacity(allocator, 8);
        defer buf.deinit();

        const n_pieces = c.llama_token_to_piece(model, new_token_id, buf.items.ptr, @intCast(buf.items.len), true);
        if (n_pieces < 0) {
            try buf.resize(@intCast(-n_pieces));
            const check = c.llama_token_to_piece(model, new_token_id, buf.items.ptr, @intCast(buf.items.len), true);
            std.debug.assert(check == -n_pieces);
        } else {
            try buf.resize(@intCast(n_pieces));
        }
        try console.print("{s}", .{buf.items});

        batch.token[0] = new_token_id;
        batch.pos[0] = n_cur;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = @intFromBool(true);

        batch.n_tokens = 1;

        n_cur += 1;

        const new_decode_res = c.llama_decode(ctx, batch);
        if (new_decode_res != 0) {
            std.debug.print("Failed to decode: {d}\n", .{new_decode_res});
            return;
        }
    }

    c.llama_print_timings(ctx);

    c.llama_batch_free(batch);
    c.llama_free(ctx);
    c.llama_free_model(model);
    c.llama_backend_free();
}

// test "simple test" {
//     var list = std.ArrayList(i32).init(std.testing.allocator);
//     defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
//     try list.append(42);
//     try std.testing.expectEqual(@as(i32, 42), list.pop());
// }
