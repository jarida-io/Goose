# PCTX Code Execution Runtime

A Deno extension providing:
- **MCP Client**: Model Context Protocol client functionality
- **JS Local Tools**: JavaScript callback-based tools with dependency support
- **Console Capturing**: Automatic stdout/stderr capture
- **Network Permissions**: Host-based fetch controls

## Primary Use Case: TypeScript SDK with Dependencies

The main use case is running PCTX as a TypeScript SDK where users define tools with their own dependencies:

```javascript
// User has Zod in their environment
import { z } from 'npm:zod';

// Define a tool that uses Zod
registerJsLocalTool({
    name: "createUser",
    description: "Creates a user with validation"
}, (args) => {
    const UserSchema = z.object({
        name: z.string().min(2),
        email: z.string().email(),
        age: z.number().min(0).max(120)
    });
    return UserSchema.parse(args); // Uses Zod for validation!
});

// Sandboxed code calls the tool (doesn't need Zod directly)
const user = await callJsLocalTool("createUser", {
    name: "Alice",
    email: "alice@example.com",
    age: 30
});
```

**Key benefit**: Dependencies (Zod, database clients, APIs, etc.) are available where tools are **defined** (trusted zone), but sandboxed code just calls the tools without needing dependency access.

See the complete example: [typescript_sdk_with_dependencies.rs](examples/typescript_sdk_with_dependencies.rs)

## Quick Start

```rust
use deno_core::{JsRuntime, RuntimeOptions};
use pctx_code_execution_runtime::{
    pctx_runtime_snapshot, MCPRegistry, JsCallableToolRegistry,
    AllowedHosts, RUNTIME_SNAPSHOT
};

// Create registries
let mcp_registry = MCPRegistry::new();
let js_local_tool_registry = JsCallableToolRegistry::new();
let allowed_hosts = AllowedHosts::new(Some(vec!["example.com".to_string()]));

let mut runtime = JsRuntime::new(RuntimeOptions {
    startup_snapshot: Some(RUNTIME_SNAPSHOT),
    extensions: vec![pctx_runtime_snapshot::init(
        mcp_registry,
        js_local_tool_registry,
        allowed_hosts
    )],
    ..Default::default()
});

// MCP API and JS Local Tools are now available in JavaScript
let code = r#"
    // Register an MCP server
    registerMCP({ name: "my-server", url: "http://localhost:3000" });

    // Register a local tool with a callback
    registerJsLocalTool({
        name: "calculator",
        description: "Performs arithmetic",
    }, (args) => {
        return args.a + args.b;
    });

    // Call MCP tool
    const mcpResult = await callMCPTool({
        name: "my-server",
        tool: "get_data",
        arguments: { id: 42 }
    });

    // Call local tool
    const localResult = await callJsLocalTool("calculator", { a: 10, b: 5 });

    console.log("MCP:", mcpResult, "Local:", localResult);
"#;

runtime.execute_script("<main>", code)?;
```

## Rust API Reference

### Core Types

#### `MCPRegistry`

Thread-safe registry for MCP server configurations.

```rust
let registry = MCPRegistry::new();
```

#### `JsCallableToolRegistry`

Thread-safe registry for local tool metadata (callbacks stored in JS).

```rust
let callable_registry = JsCallableToolRegistry::new();

// Query tools
if callable_registry.has("my-tool") {
    println!("Tool exists!");
}

let metadata = callable_registry.get_metadata("my-tool");
let all_tools = callable_registry.list();
```

#### `AllowedHosts`

Whitelist of hosts allowed for network access.

```rust
let allowed_hosts = AllowedHosts::new(Some(vec![
    "example.com".to_string(),
    "api.service.com".to_string(),
]));
```

### Snapshot

#### `RUNTIME_SNAPSHOT`

Pre-compiled V8 snapshot containing the runtime.

```rust
pub static RUNTIME_SNAPSHOT: &[u8] = /* ... */;
```

## Examples

### Console Output Capture

```rust
let code = r#"
    console.log("Line 1");
    console.log("Line 2");
    console.error("Error line");

    export default {
        stdout: globalThis.__stdout,
        stderr: globalThis.__stderr
    };
"#;

let result = runtime.execute_script("<capture>", code)?;

// Extract captured output
let scope = &mut runtime.handle_scope();
let local = v8::Local::new(scope, result);
let output = serde_v8::from_v8::<serde_json::Value>(scope, local)?;

println!("Stdout: {:?}", output["stdout"]);
println!("Stderr: {:?}", output["stderr"]);
```

### Network Permissions

```rust
// Allow only specific hosts
let allowed_hosts = AllowedHosts::new(Some(vec![
    "api.example.com".to_string(),
    "cdn.example.com".to_string(),
]));

let mut runtime = JsRuntime::new(RuntimeOptions {
    startup_snapshot: Some(RUNTIME_SNAPSHOT),
    extensions: vec![pctx_runtime_snapshot::init(
        MCPRegistry::new(),
        allowed_hosts
    )],
    ..Default::default()
});

let code = r#"
    // This will succeed
    await fetch("http://api.example.com/data");

    // This will fail - host not allowed
    try {
        await fetch("http://malicious.com/data");
    } catch (e) {
        console.error("Blocked:", e.message);
    }
"#;

runtime.execute_script("<permissions>", code)?;
```

## Security

### Network Access

- Only whitelisted hosts can be accessed via `fetch()`
- Attempts to access non-whitelisted hosts throw errors
- Host matching is exact (no wildcards)

### MCP Registry

- Each runtime instance has its own isolated registry
- No cross-runtime access to MCP configurations
- Registry is not persisted between runtime sessions

### Console Capture

- Captured output is stored in runtime-local buffers
- No disk I/O or external logging
- Buffers cleared when runtime is dropped

## Performance

- **Startup**: Instant (V8 snapshot loads in <1ms)
- **Memory**: ~2MB base runtime overhead
- **MCP Operations**: Native Rust performance
- **Console Capture**: Minimal overhead (~1% per log)

## License

MIT

## Contributing

Contributions welcome! Please ensure:

- All tests pass: `cargo test --package pctx_runtime`
- Code is formatted: `cargo fmt`
- Documentation is updated

## Features

### JS Local Tools

Local tools allow you to define JavaScript callbacks that can be invoked from sandboxed code. See [JS_LOCAL_TOOLS.md](JS_LOCAL_TOOLS.md) for comprehensive documentation.

**Quick example:**

```javascript
// Register a tool with a callback
registerJsLocalTool({
    name: "file-reader",
    description: "Reads a file from the host system",
    inputSchema: {
        type: "object",
        properties: {
            path: { type: "string" }
        }
    }
}, async (args) => {
    const fs = require('fs').promises;
    return await fs.readFile(args.path, 'utf8');
});

// Call the tool from sandboxed code
const content = await callJsLocalTool("file-reader", { path: "./data.txt" });
```

**Benefits:**
- Fast: No IPC overhead
- Simple: Just JavaScript callbacks
- Flexible: Sync or async
- Type-safe: Full TypeScript support

### MCP Client

Connect to external MCP servers for tool integration.

```javascript
registerMCP({
    name: "github",
    url: "http://localhost:3000"
});

const result = await callMCPTool({
    name: "github",
    tool: "create_issue",
    arguments: { title: "Bug report" }
});
```

## Examples

See the [examples/](examples/) directory:
- **[typescript_sdk_with_dependencies.rs](examples/typescript_sdk_with_dependencies.rs)** - **Primary use case**: Shows how TypeScript SDK users can define tools with custom dependencies (Zod, DB clients, etc.) and use them from sandboxed code

## See Also

- [`pctx_type_check`](../pctx_type_check) - TypeScript type checking runtime
- [`pctx_executor`](../pctx_executor) - Complete TypeScript execution environment
- [Model Context Protocol](https://modelcontextprotocol.io) - MCP specification
- [JS_LOCAL_TOOLS.md](JS_LOCAL_TOOLS.md) - Detailed JS local tools documentation
