# MCP and optional code-mode security dependencies

This branch ports aaif-goose/goose commit
`7b7b8aa58f54a7e189f681883de55bd42b633634` onto the GIAP fork and pins RMCP
2.1.0. This includes the fixes for GHSA-33f5-2c5q-wgwj,
GHSA-9pj6-vhgr-3mwh, GHSA-c9xm-49cp-xcr9 and GHSA-9g45-5xwm-f3wc.
The fork-specific LiteRT tool-result conversion also uses ContentBlock.
Custom HTTP clients, including OAuth and PCTX clients, explicitly disable
redirects because supplying a client bypasses RMCP's default redirect policy.

Optional code-mode remains available. Four small compatibility patches to
published PCTX crates remove its RMCP 1.x and OpenTelemetry SDK 0.31 instances.
The SDK now resolves to 0.32.1. Source versions, commits, licenses and the
patch boundaries are recorded in vendor/SECURITY-PATCHES.md.

Parent workspaces must mirror these patches and the RMCP pin. GIAP's matching
PR adapts its content types and checks out this exact submodule commit in CI.
Do not move the fork main underneath a parent that still requires RMCP 1.x;
merge the coordinated dependency and integration PRs together.

Local validation in GIAP: production server and adapter all-target compilation,
optional code-mode/rustls compilation, 442 provider-model tests, 271 adapter
unit tests, and the MCP server suite. Some existing tests are explicitly
ignored; no live-model or external OAuth integration result is claimed.
Jetson GPU inference, third-party MCP servers and HawkScan DAST remain
unverified. HawkScan is unavailable without its CLI and API key.
